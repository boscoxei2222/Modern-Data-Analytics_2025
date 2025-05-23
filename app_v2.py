from shiny import App, ui, reactive, render
import pandas as pd
import ast
from recommender.matching import get_top_matches
from ipyleaflet import Map, Marker, Popup, Icon, TileLayer
import matplotlib.pyplot as plt
import seaborn as sns
from shinywidgets import output_widget, render_widget  
from ipywidgets import HTML
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


#Load project data
project_data = pd.read_csv("data_1/processed/project_merged.csv")
project_data["sciVocTopics"] = project_data["sciVocTopics"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)
mlb = MultiLabelBinarizer()
scivoc_matrix = mlb.fit_transform(project_data["sciVocTopics"])
scivoc_df = pd.DataFrame(scivoc_matrix, columns=mlb.classes_)
project_data = pd.concat([project_data, scivoc_df.add_prefix("SCV_")], axis=1)
org_data = pd.read_csv("data_1/processed/org_unique_detailed.csv")

# Define text_vector globally
text_vector = None  # Placeholder for text vector

# UI
app_ui = ui.page_fluid(
    ui.navset_pill(  
        ui.nav_panel("Proposal Match",
                     ui.layout_columns(
                        ui.card(
                             ui.input_text_area("proposal", "Enter your research proposal:", rows=6),
                             ui.input_slider("top_n", "Number of results:", min=10, max=20, value=10),
                             ui.input_slider("scivoc_weight", "EuroSciVoc Weight:", min=0, max=1, value=0.5, step=0.1),
                             ui.h6("📅 Filter by Project Year", class_="mt-3"),
                             ui.input_slider(
                                 id="year_range",
                                 label="Select Years:",
                                 min=2021,
                                 max=2026,
                                 value=(2024, 2026),
                                 step=1,
                                 ticks=str([2021, 2023, 2026]),
                                 width="100%"
                             ),
                             ui.output_text_verbatim("selected_years"),
                             ui.tooltip(
                                 ui.span("💡 Tip: Drag to select time range", class_="text-muted"),
                                 "Default shows projects from the last 3 years, drag to adjust"
                             ),
                             ui.input_action_button("submit", "Find Matching Projects")),
                        ui.card(
                            ui.output_table("match_summary"),
                            ui.output_plot("similarity_plot"),
                            ui.output_plot("funding_scheme_plot")
                        )
                        )
                     ),

        ui.nav_panel("Project Summaries", 
                     ui.layout_columns(
                         ui.card(
                             ui.output_ui("acronym_list"),
                             ui.output_ui("project_detail")
                            ),
                         ui.accordion(
                             ui.accordion_panel("Organisations Overview",
                                 output_widget("map"),
                                 ui.output_table("org_summary")
                             ),
                             ui.accordion_panel("Funding Overview",
                                 ui.output_ui("funding_summary")
                            )
                            )
                        
                        )
                    ),
        ui.nav_panel("Organisation Profile",
                     ui.layout_columns(
                         # 第一屏：关键指标卡片
                         ui.card(
                             ui.input_select("org_name", "Select an organization:", 
                                           choices=sorted(org_data["name"].unique().tolist())),
                             ui.output_ui("org_basic_info"),
                             ui.layout_columns(
                                 ui.value_box(
                                     "Total Funding",
                                     ui.output_ui("total_funding"),
                                     showcase=ui.span("€", style="font-size: 2rem;")
                                 ),
                                 ui.value_box(
                                     "Active Projects",
                                     ui.output_ui("active_projects"),
                                     showcase=ui.span("📊", style="font-size: 2rem;")
                                 ),
                                 col_widths=[6, 6]
                             )
                         ),
                         # 第二屏：地理分布与合作网络
                         ui.card(
                             ui.h5("Geographic Distribution"),
                             output_widget("org_map"),
                             ui.h5("Collaboration Network"),
                             ui.output_table("org_projects")
                         )
                     )
                    ),


        ui.nav_panel("Funding Mechanisms",
                     ui.card(
                         ui.output_plot("pie_topic"),
                         ui.output_ui("funding_list"),
                         ui.output_ui("funding_detail")
                    )
        ),
                    
        id="tab",
    )  
)


# Server
def server(input, output, session):
    
    # Reactive value to hold the match results
    matches = reactive.Value(pd.DataFrame())

    # Preprocess EuroSciVoc data
    def preprocess_scivoc_data(df):
        scivoc_data = df['sciVocTopics'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        mlb = MultiLabelBinarizer()
        scivoc_matrix = mlb.fit_transform(scivoc_data)
        return scivoc_matrix, mlb.classes_

    # When user clicks the button, update matches
    @reactive.effect
    @reactive.event(input.submit)
    def update_matches():
        proposal = input.proposal()
        if not proposal.strip():
            matches.set(pd.DataFrame())  # empty input
            return

        # Get base matching results
        top_match_ids_scores, _ = get_top_matches(
            proposal_text=proposal,
            project_data=project_data,
            top_n=input.top_n(),
            scivoc_weight=input.scivoc_weight()
        )
        
        ids = [pid for pid, _ in top_match_ids_scores]
        scores = {pid: score for pid, score in top_match_ids_scores}

        match_df = project_data[project_data["projectID"].isin(ids)].copy()
        match_df["similarity"] = match_df["projectID"].map(scores)
        
        # === New filtering logic ===
        # Year filtering
        match_df = match_df[
            (match_df['startDate'].str[:4].astype(int) >= input.year_range()[0]) & 
            (match_df['startDate'].str[:4].astype(int) <= input.year_range()[1])
        ]
        
        match_df.sort_values("similarity", ascending=False, inplace=True)
        matches.set(match_df)

    @render.text
    def selected_years():
        return f"Showing projects from {input.year_range()[0]} to {input.year_range()[1]}"

    # helper function to get project organisations (used in map rendering)
    def get_project_orgs(acronym):
        df = matches.get()
        orgs = []

        if not acronym:
            return pd.DataFrame()

        row = df[df["acronym"] == acronym].iloc[0]

        role_columns = ["coordinator", "participant", "thirdParty", "associatedPartner"]

        for role in role_columns:
            
            raw = row.get(role, [])
            if isinstance(raw, str):
                try:
                    ids = ast.literal_eval(raw)
                except Exception:
                    ids = []
            else:
                ids = raw
            
            if isinstance(ids, list):
                for org_id in ids:
                    orgs.append({"organisationID": org_id, "role": role})
        org_df = pd.DataFrame(orgs)

        # Merge with org_data to get name and location
        result = org_df.merge(org_data, on="organisationID", how="left")
        return result

    # Output the project match summary (Acronym & Title)
    @render.table
    def match_summary():
        df = matches.get()
        if df.empty:
            return pd.DataFrame({"Similar Projects": ["No results yet. Please enter a proposal."]})
        
        # Format similarity scores as percentages
        df['similarity'] = df['similarity'].apply(lambda x: f"{x*100:.1f}%")
        
        # Return table with similarity scores
        return df[["acronym", "title", "similarity"]].copy()

    # Add similarity distribution plot
    @render.plot
    def similarity_plot():
        df = matches.get()
        if df.empty:
            return None
        
        plt.figure(figsize=(8, 4))
        sns.barplot(data=df, x='acronym', y='similarity')
        plt.xticks(rotation=45)
        plt.title('Similarity Scores Distribution')
        plt.xlabel('Project Acronym')
        plt.ylabel('Similarity Score')
        plt.tight_layout()
        return plt.gcf()

    # 添加 funding scheme 分布图
    @render.plot
    def funding_scheme_plot():
        df = matches.get()
        if df.empty:
            return None
        
        # 计算 funding scheme 的分布
        funding_counts = df['fundingScheme'].value_counts()
        
        plt.figure(figsize=(8, 4))
        funding_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Funding Scheme Distribution')
        plt.ylabel('')
        plt.tight_layout()
        return plt.gcf()

    # Output the acronym list
    @render.ui
    def acronym_list():
        df = matches.get()
        if df.empty or "acronym" not in df.columns:
            return ui.p("No results yet.")

        options = df["acronym"].dropna().unique().tolist()
        return ui.input_select("selected_project", "Select a project acronym:", choices=options)
    
    # Output the project detail
    @render.ui
    def project_detail():
        df = matches.get()
        selected = input.selected_project()
        if not selected:
            return ui.p("Select a project to view details.")

        row = df[df["acronym"] == selected].iloc[0]
        
        # Get active EuroSciVoc topics from global encoding
        active_scivoc = [col.replace("SCV_", "") for col in project_data.columns 
                        if col.startswith("SCV_") and row[col] == 1]

        return ui.panel_well(
            ui.h4(row["title"]),
            ui.p(f"Objective: {row['objective']}"),
            ui.p(f"EuroSciVoc Topics: {', '.join(active_scivoc)}"),
            ui.a("View full project on CORDIS", href=row["cordis_project_url"], target="_blank")
        )

    # Output the map
    @render_widget  
    def map():
        acronym = input.selected_project()

        if not acronym:
            m = Map(center=(50, 10), zoom=4, scroll_wheel_zoom=True)
            m.add_layer(TileLayer(url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'))
            return m

        orgs = get_project_orgs(acronym)
        if orgs.empty:
            m = Map(center=(50, 10), zoom=4, scroll_wheel_zoom=True)
            m.add_layer(TileLayer(url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'))
            return m

        m = Map(center=(50, 10), zoom=4, scroll_wheel_zoom=True)
        m.add_layer(TileLayer(url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'))
        
        # 使用默认图标
        default_icon = Icon(
            icon_url='https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
            icon_size=[25, 41],
            icon_anchor=[12, 41]
        )
        
        coordinator_icon = Icon(
            icon_url='https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
            icon_size=[25, 41],
            icon_anchor=[12, 41]
        )

        for _, row in orgs.iterrows():
            if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
                continue
            
            # 根据角色选择图标
            icon = coordinator_icon if row['role'] == "coordinator" else default_icon
            
            marker = Marker(
                icon=icon,
                location=(row["latitude"], row["longitude"]),
                title=f"{row['name']} ({row['role']})",
                draggable=False
            )
            
            # 添加弹出信息
            popup_content = f"""
            <div style='font-family: Arial, sans-serif; padding: 10px;'>
                <h4 style='margin: 0 0 5px 0;'>{row['name']}</h4>
                <p style='margin: 0;'><strong>Role:</strong> {row['role']}</p>
                <p style='margin: 0;'><strong>Country:</strong> {row['country']}</p>
            </div>
            """
            marker.popup = HTML(popup_content)
            
            m.add(marker)

        return m

    @render.table
    def org_summary():
        acronym = input.selected_project()
        if not acronym:
            return ui.p("No project selected.")

        df = get_project_orgs(acronym)
        if df.empty:
            return ui.p("No organisations found.")

        return df[["name", "role", "country"]].copy().reset_index(drop=True)

    # Output the project funding summary
    @render.ui
    def funding_summary():
        df = matches.get()
        selected = input.selected_project()
        if not selected:
            return ui.p("Select a project to view details.")

        row = df[df["acronym"] == selected].iloc[0]

        return ui.panel_well(
            ui.p(f"Total Funding: €{row['ecMaxContribution']:,.0f}"),
            ui.p(f"Average Annual Funding per Participant: €{row['avg_annual_funding_per_participant']:,.0f}"),
            ui.p(f"Funding Scheme: {row['funding_id']}"),
            ui.p(f"{row['title_topic']}")
        )

    # Output the boxplot
    @render.plot
    def boxplot_funding():
        df = matches.get()
        if df.empty:
            return

        plt.figure(figsize=(6, 2))
        sns.boxplot(x=df["ecMaxContribution"]/1e6)
        plt.title("Project Funding")
        plt.xlabel("Funding in millions€")
        return plt.gcf()

    # Output the pie chart
    @render.plot
    def pie_topic():
        df = matches.get()
        if df.empty or "title_topic" not in df.columns:
            return

        topic_counts = df["title_topic"].value_counts()
        plt.figure(figsize=(6, 6))
        topic_counts.plot.pie(startangle=90, autopct='%1.1f%%', textprops={'fontsize': 10})
        plt.ylabel("")
        plt.title("Grants awarded by")
        return plt.gcf()

    # Output the acronym list
    @render.ui
    def funding_list():
        df = matches.get()
        if df.empty or "title_topic" not in df.columns:
            return ui.p("No results yet.")

        options = df["title_topic"].dropna().unique().tolist()

        return ui.input_select("selected_funding", "Select a funding scheme:",
                                         choices=options)

    # Output the funding detail
    @render.ui
    def funding_detail():
        df = matches.get()
        selected = input.selected_funding()
        if not selected:
            return ui.p("Select a funding scheme to view details.")

        row = df[df["title_topic"] == selected].iloc[0]

        return ui.panel_well(
            ui.h4(row["title_topic"]),
            ui.HTML(row['topic_objective']),
            ui.p(""),
            ui.a("View full project on CORDIS", href=row["cordis_funding_url"], target="_blank")

    
        )

    # Organization Profile 相关函数
    @render.ui
    def org_basic_info():
        selected_org = input.org_name()
        if not selected_org:
            return ui.p("Please select an organization.")
        
        org = org_data[org_data["name"] == selected_org].iloc[0]
        
        return ui.panel_well(
            ui.h4(selected_org),
            ui.p(f"Organization Type: {org.get('organizationType', 'N/A')}")
        )

    @render.ui
    def total_funding():
        selected_org = input.org_name()
        if not selected_org:
            return "0"
        
        org_id = org_data[org_data["name"] == selected_org]["organisationID"].iloc[0]
        org_projects = project_data[
            project_data.apply(lambda x: any(
                org_id in (ast.literal_eval(x[role]) if isinstance(x[role], str) else x[role])
                for role in ["coordinator", "participant", "thirdParty", "associatedPartner"]
            ), axis=1)
        ]
        
        if org_projects.empty:
            return "0"
        
        total = org_projects["ecMaxContribution"].sum()
        return f"{total:,.0f}"

    @render.ui
    def active_projects():
        selected_org = input.org_name()
        if not selected_org:
            return "0"
        
        org_id = org_data[org_data["name"] == selected_org]["organisationID"].iloc[0]
        org_projects = project_data[
            project_data.apply(lambda x: any(
                org_id in (ast.literal_eval(x[role]) if isinstance(x[role], str) else x[role])
                for role in ["coordinator", "participant", "thirdParty", "associatedPartner"]
            ), axis=1)
        ]
        
        return str(len(org_projects))

    @render_widget
    def org_map():
        selected_org = input.org_name()
        if not selected_org:
            m = Map(center=(50, 10), zoom=4, scroll_wheel_zoom=True)
            m.add_layer(TileLayer(url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'))
            return m
        
        # Get selected organization information
        org = org_data[org_data["name"] == selected_org].iloc[0]
        
        # Create map
        m = Map(center=(50, 10), zoom=4, scroll_wheel_zoom=True)
        m.add_layer(TileLayer(url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'))
        
        # Check if organization has location information
        if pd.notna(org.get("latitude")) and pd.notna(org.get("longitude")):
            try:
                lat = float(org["latitude"])
                lon = float(org["longitude"])
                
                # Create organization marker
                marker = Marker(
                    location=(lat, lon),
                    title=selected_org,
                    draggable=False
                )
                
                # Add popup information
                popup_content = f"""
                <div style='font-family: Arial, sans-serif; padding: 10px;'>
                    <h4 style='margin: 0 0 5px 0;'>{selected_org}</h4>
                    <p style='margin: 0;'><strong>Type:</strong> {org.get('organizationType', 'N/A')}</p>
                    <p style='margin: 0;'><strong>Country:</strong> {org.get('country', 'N/A')}</p>
                </div>
                """
                marker.popup = HTML(popup_content)
                m.add(marker)
                
                # Center map on organization location
                m.center = (lat, lon)
                m.zoom = 8
            except:
                pass
        
        return m

    @render.table
    def org_projects():
        selected_org = input.org_name()
        if not selected_org:
            return pd.DataFrame()
        
        org_id = org_data[org_data["name"] == selected_org]["organisationID"].iloc[0]
        org_projects = project_data[
            project_data.apply(lambda x: any(
                org_id in (ast.literal_eval(x[role]) if isinstance(x[role], str) else x[role])
                for role in ["coordinator", "participant", "thirdParty", "associatedPartner"]
            ), axis=1)
        ]
        
        if org_projects.empty:
            return pd.DataFrame()
        
        # 获取该组织在每个项目中的角色
        roles = []
        for _, row in org_projects.iterrows():
            role = None
            for r in ["coordinator", "participant", "thirdParty", "associatedPartner"]:
                if isinstance(row[r], str):
                    try:
                        ids = ast.literal_eval(row[r])
                    except:
                        ids = []
                else:
                    ids = row[r]
                
                if isinstance(ids, list) and org_id in ids:
                    role = r
                    break
            roles.append(role)
        
        org_projects = org_projects.copy()
        org_projects["role"] = roles
        
        return org_projects[["acronym", "title", "role", "fundingScheme"]].copy()

    # Add summary statistics plots in match_summary
    @render.plot
    def funding_scheme_distribution():
        df = matches.get()
        if df.empty:
            return None
        plt.figure(figsize=(8, 4))
        df['fundingScheme'].value_counts().plot(kind='bar')
        plt.title('Funding Scheme Distribution')
        plt.xlabel('Funding Scheme')
        plt.ylabel('Count')
        plt.tight_layout()
        return plt.gcf()

    @render.plot
    def similarity_score_distribution():
        df = matches.get()
        if df.empty:
            return None
        plt.figure(figsize=(8, 4))
        sns.histplot(df['similarity'], bins=10)
        plt.title('Similarity Score Distribution')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.tight_layout()
        return plt.gcf()

    @render.plot
    def partners_distribution():
        df = matches.get()
        if df.empty:
            return None
        plt.figure(figsize=(8, 4))
        sns.histplot(df['n_organisations'], bins=10)
        plt.title('Partners Distribution')
        plt.xlabel('Number of Partners')
        plt.ylabel('Count')
        plt.tight_layout()
        return plt.gcf()

    # Add funding amount distribution plot
    @render.plot
    def funding_amount_distribution():
        df = matches.get()
        if df.empty:
            return None
        plt.figure(figsize=(8, 4))
        sns.histplot(df['ecMaxContribution'], bins=10)
        plt.title('Funding Amount Distribution')
        plt.xlabel('Funding Amount (€)')
        plt.ylabel('Count')
        plt.tight_layout()
        return plt.gcf()

# App
app = App(app_ui, server) 