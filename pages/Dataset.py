import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.title("Data")
st.write("Explore and visualize the TB-DOTS CAR dataset (2015-2025).")

# Load the non-temporal dataset
@st.cache_data
def load_non_temporal_data():
    data_path = Path(__file__).parent.parent / "dataset" / "non-temporal" / "2015-2025-consolidated-clean.csv"
    df = pd.read_csv(data_path)
    return df

# Load the temporal dataset
@st.cache_data
def load_temporal_data():
    data_path = Path(__file__).parent.parent / "dataset" / "temporal" / "combined_dataset.csv"
    df = pd.read_csv(data_path)
    # Convert Age to numeric
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    # Convert monthly numeric columns
    for m in range(13):
        for sub in ['Monthly Doses Taken', 'Cumulative Doses Taken', 'Monthly Missed Doses']:
            col = f'M{m}_{sub}'
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    return df

# Helper to clean series (exclude N/A-like strings)
def clean_series(s):
    NA_STRINGS = ['N/A', 'n/a', 'NA', 'na', 'None', 'none', 'N/a', 'nan']
    s = s.astype(str).str.strip()
    s = s.replace(NA_STRINGS, np.nan)
    s = s.replace('nan', np.nan)
    return s

tab1, tab2 = st.tabs(["Temporal", "Non Temporal"])

with tab1:
    st.subheader("Temporal Data Visualization")
    st.write("TB patient records from 4 health facilities in Baguio City (2016–2025)")
    
    try:
        df_temp = load_temporal_data()
        
        # Summary Statistics
        st.markdown("### 📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", f"{len(df_temp):,}")
        with col2:
            st.metric("Facilities", f"{df_temp['Facility'].nunique()}")
        with col3:
            year_range = f"{int(df_temp['Data_Year'].min())} - {int(df_temp['Data_Year'].max())}"
            st.metric("Year Range", year_range)
        with col4:
            avg_age = df_temp['Age'].mean()
            st.metric("Average Age", f"{avg_age:.1f} years")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            male_pct = (clean_series(df_temp['Sex']) == 'Male').mean() * 100
            st.metric("Male %", f"{male_pct:.1f}%")
        with col2:
            outcome_clean = clean_series(df_temp['Outcome'])
            cured_pct = outcome_clean.str.lower().str.contains('cured|completed', na=False).mean() * 100
            st.metric("Treatment Success", f"{cured_pct:.1f}%")
        with col3:
            died_pct = outcome_clean.str.lower().str.contains('died', na=False).mean() * 100
            st.metric("Mortality Rate", f"{died_pct:.1f}%")
        with col4:
            bact_confirmed = clean_series(df_temp['Bacteriologic Status']).str.lower().str.contains('bacteriolog', na=False).mean() * 100
            st.metric("Bacteriologically Confirmed", f"{bact_confirmed:.1f}%")
        
        st.divider()
        
        # Cases by Year and Facility
        st.markdown("## 📅 Temporal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Cases by Year")
            cases_year = df_temp.groupby('Data_Year').size().reset_index(name='Count')
            fig_year = px.bar(
                cases_year,
                x='Data_Year',
                y='Count',
                labels={'Data_Year': 'Year', 'Count': 'Number of Cases'},
                color_discrete_sequence=['steelblue']
            )
            fig_year.update_layout(height=350)
            st.plotly_chart(fig_year, use_container_width=True)
        
        with col2:
            st.markdown("### Cases by Facility")
            cases_fac = df_temp['Facility'].value_counts().reset_index()
            cases_fac.columns = ['Facility', 'Count']
            fig_fac = px.pie(
                cases_fac,
                values='Count',
                names='Facility',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_fac.update_layout(height=350)
            st.plotly_chart(fig_fac, use_container_width=True)
        
        # Cases per Year by Facility
        st.markdown("### Cases per Year by Facility")
        cases_yf = df_temp.groupby(['Data_Year', 'Facility']).size().reset_index(name='Count')
        fig_yf = px.bar(
            cases_yf,
            x='Data_Year',
            y='Count',
            color='Facility',
            barmode='group',
            labels={'Data_Year': 'Year', 'Count': 'Number of Cases'}
        )
        fig_yf.update_layout(height=400)
        st.plotly_chart(fig_yf, use_container_width=True)
        
        st.divider()
        
        # Demographic Analysis
        st.markdown("## 👥 Demographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Age Distribution")
            valid_ages = df_temp[(df_temp['Age'] >= 0) & (df_temp['Age'] <= 120)]['Age']
            fig_age = px.histogram(
                valid_ages,
                nbins=20,
                labels={'value': 'Age', 'count': 'Count'},
                color_discrete_sequence=['steelblue']
            )
            fig_age.add_vline(x=valid_ages.median(), line_dash="dash", line_color="red",
                             annotation_text=f"Median: {valid_ages.median():.0f}")
            fig_age.update_layout(height=350, showlegend=False, xaxis_title="Age", yaxis_title="Count")
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.markdown("### Sex Distribution")
            sex_clean = clean_series(df_temp['Sex']).dropna()
            sex_counts = sex_clean.value_counts().reset_index()
            sex_counts.columns = ['Sex', 'Count']
            fig_sex = px.pie(
                sex_counts,
                values='Count',
                names='Sex',
                color_discrete_sequence=['#3498db', '#e74c3c']
            )
            fig_sex.update_layout(height=350)
            st.plotly_chart(fig_sex, use_container_width=True)
        
        # Age by Facility
        st.markdown("### Age Distribution by Facility")
        fig_age_fac = px.box(
            df_temp[(df_temp['Age'] >= 0) & (df_temp['Age'] <= 120)],
            x='Facility',
            y='Age',
            color='Facility',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_age_fac.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_age_fac, use_container_width=True)
        
        st.divider()
        
        # Clinical Characteristics
        st.markdown("## 🏥 Clinical Characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Treatment Outcomes")
            outcome_clean = clean_series(df_temp['Outcome']).dropna()
            outcome_counts = outcome_clean.value_counts().reset_index()
            outcome_counts.columns = ['Outcome', 'Count']
            fig_outcome = px.bar(
                outcome_counts.head(10),
                x='Count',
                y='Outcome',
                orientation='h',
                color_discrete_sequence=['#17becf']
            )
            fig_outcome.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_outcome, use_container_width=True)
        
        with col2:
            st.markdown("### Case Registration Group")
            crg_clean = clean_series(df_temp['Case Registration Group']).dropna()
            crg_counts = crg_clean.value_counts().reset_index()
            crg_counts.columns = ['Group', 'Count']
            fig_crg = px.bar(
                crg_counts,
                x='Count',
                y='Group',
                orientation='h',
                color_discrete_sequence=['#e377c2']
            )
            fig_crg.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_crg, use_container_width=True)
        
        # Outcome by Facility
        st.markdown("### Treatment Outcomes by Facility")
        df_out_fac = df_temp.copy()
        df_out_fac['Outcome_clean'] = clean_series(df_temp['Outcome'])
        valid_out = df_out_fac.dropna(subset=['Outcome_clean'])
        out_fac = valid_out.groupby(['Facility', 'Outcome_clean']).size().reset_index(name='Count')
        # Filter to top 5 outcomes
        top_outcomes = valid_out['Outcome_clean'].value_counts().head(5).index.tolist()
        out_fac_filtered = out_fac[out_fac['Outcome_clean'].isin(top_outcomes)]
        fig_out_fac = px.bar(
            out_fac_filtered,
            x='Facility',
            y='Count',
            color='Outcome_clean',
            barmode='group',
            labels={'Outcome_clean': 'Outcome'}
        )
        fig_out_fac.update_layout(height=400)
        st.plotly_chart(fig_out_fac, use_container_width=True)
        
        st.divider()
        
        # Treatment Adherence Analysis
        st.markdown("## 💊 Treatment Adherence Analysis")
        
        # Monthly Doses Taken
        dose_cols = [f'M{m}_Monthly Doses Taken' for m in range(13) if f'M{m}_Monthly Doses Taken' in df_temp.columns]
        if dose_cols:
            mean_doses = []
            months = []
            for col in dose_cols:
                val = pd.to_numeric(df_temp[col], errors='coerce').mean()
                if not pd.isna(val):
                    mean_doses.append(val)
                    months.append(col.replace('_Monthly Doses Taken', ''))
            
            if mean_doses:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Average Monthly Doses Taken")
                    dose_df = pd.DataFrame({'Month': months, 'Mean Doses': mean_doses})
                    fig_doses = px.bar(
                        dose_df,
                        x='Month',
                        y='Mean Doses',
                        color_discrete_sequence=['coral']
                    )
                    fig_doses.update_layout(height=350)
                    st.plotly_chart(fig_doses, use_container_width=True)
                
                with col2:
                    st.markdown("### Cumulative Doses Over Treatment")
                    cumul_cols = [f'M{m}_Cumulative Doses Taken' for m in range(13) if f'M{m}_Cumulative Doses Taken' in df_temp.columns]
                    mean_cumul = []
                    months_c = []
                    for col in cumul_cols:
                        val = pd.to_numeric(df_temp[col], errors='coerce').mean()
                        if not pd.isna(val):
                            mean_cumul.append(val)
                            months_c.append(col.replace('_Cumulative Doses Taken', ''))
                    
                    if mean_cumul:
                        cumul_df = pd.DataFrame({'Month': months_c, 'Mean Cumulative': mean_cumul})
                        fig_cumul = px.line(
                            cumul_df,
                            x='Month',
                            y='Mean Cumulative',
                            markers=True,
                            color_discrete_sequence=['steelblue']
                        )
                        fig_cumul.update_layout(height=350)
                        st.plotly_chart(fig_cumul, use_container_width=True)
        
        st.divider()
        
        # Co-morbidities & Drug Resistance
        st.markdown("## 🩺 Co-morbidities & Drug Resistance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top Co-morbidities")
            if 'Co-morbidities' in df_temp.columns:
                como = clean_series(df_temp['Co-morbidities']).dropna()
                como_counts = como.value_counts().head(10).reset_index()
                como_counts.columns = ['Co-morbidity', 'Count']
                fig_como = px.bar(
                    como_counts,
                    x='Count',
                    y='Co-morbidity',
                    orientation='h',
                    color_discrete_sequence=['mediumpurple']
                )
                fig_como.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_como, use_container_width=True)
        
        with col2:
            st.markdown("### Drug Resistance Status")
            if 'Drug Resistance Bacteriological Status' in df_temp.columns:
                dr = clean_series(df_temp['Drug Resistance Bacteriological Status']).dropna()
                dr_counts = dr.value_counts().reset_index()
                dr_counts.columns = ['Status', 'Count']
                fig_dr = px.bar(
                    dr_counts,
                    x='Count',
                    y='Status',
                    orientation='h',
                    color_discrete_sequence=['darkorange']
                )
                fig_dr.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_dr, use_container_width=True)
        
        st.divider()
        
        # Data Preview
        st.markdown("## 📄 Data Preview")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            year_filter_t = st.multiselect("Filter by Year", 
                                           options=sorted(df_temp['Data_Year'].dropna().unique().astype(int)), 
                                           default=[], key='temporal_year')
        with col2:
            facility_filter = st.multiselect("Filter by Facility", 
                                             options=df_temp['Facility'].unique().tolist(), 
                                             default=[], key='temporal_facility')
        with col3:
            outcome_options = clean_series(df_temp['Outcome']).dropna().unique().tolist()
            outcome_filter_t = st.multiselect("Filter by Outcome", 
                                              options=outcome_options, 
                                              default=[], key='temporal_outcome')
        
        # Apply filters
        filtered_temp = df_temp.copy()
        if year_filter_t:
            filtered_temp = filtered_temp[filtered_temp['Data_Year'].isin(year_filter_t)]
        if facility_filter:
            filtered_temp = filtered_temp[filtered_temp['Facility'].isin(facility_filter)]
        if outcome_filter_t:
            filtered_temp = filtered_temp[clean_series(filtered_temp['Outcome']).isin(outcome_filter_t)]
        
        st.write(f"Showing {len(filtered_temp):,} records")
        
        # Select columns to display (exclude monthly columns for cleaner view)
        display_cols = [c for c in filtered_temp.columns if not c.startswith('M') or c in ['M0_Weight', 'M0_Height']]
        st.dataframe(filtered_temp[display_cols], use_container_width=True, height=400)
        
    except FileNotFoundError:
        st.error("Temporal dataset not found. Please ensure 'combined_dataset.csv' exists in dataset/temporal/ folder.")
    except Exception as e:
        st.error(f"Error loading temporal data: {str(e)}")

with tab2:
    st.subheader("Non-Temporal Data Visualization")
    
    # Load data
    try:
        df = load_non_temporal_data()
        
        # Mapping dictionaries for categorical values
        sex_map = {"M": "Male", "F": "Female"}
        outcome_map = {
            "CURED": "Cured", 
            "TREATMENT COMPLETED": "Treatment Completed", 
            "DIED": "Died", 
            "TREATMENT FAILED": "Treatment Failed",
            "LOST TO FF-UP": "Lost to Follow-up", 
            "NOT EVALUATED": "Not Evaluated", 
            "STILL ON TREATMENT": "Still on Treatment",
            "TRANSFERRED OUT": "Transferred Out"
        }
        anatomical_site_map = {"EP": "Extra-pulmonary", "P": "Pulmonary"}
        bacteriologic_map = {
            "Clinically-diagnosed TB": "Clinically Diagnosed", 
            "Bacteriologically-confirmed TB": "Bacteriologically Confirmed"
        }
        registration_map = {
            "NEW": "New", 
            "RELAPSE": "Relapse", 
            "TREATMENT AFTER FAILURE": "Treatment After Failure",
            "TREATMENT AFTER LTFU": "Treatment After LTFU", 
            "OTHER PREVIOUSLY TREATED": "Other Previously Treated", 
            "UNKNOWN PREVIOUS TB TREATMENT": "Unknown Previous TB Treatment"
        }
        source_map = {
            "COMMUNITY": "Community",
            "CONTACT INVESTIGATION": "Contact Investigation",
            "OTHER PUBLIC FACILITY": "Other Public Facility",
            "PRIVATE FACILITY": "Private Facility",
            "PUBLIC HEALTH CENTER": "Public Health Center"
        }
        
        # Summary Statistics
        st.markdown("### 📊 Summary Statistics")
        
        # Row 1: Basic counts
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Year Range", f"{int(df['Year'].min())} - {int(df['Year'].max())}")
        with col3:
            avg_age = df[(df['Age'] >= 0) & (df['Age'] <= 120)]['Age'].mean()
            st.metric("Average Age", f"{avg_age:.1f} years")
        with col4:
            male_pct = (df['Sex'] == 'M').mean() * 100
            st.metric("Male %", f"{male_pct:.1f}%")
        
        # Row 2: Clinical statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pulmonary_pct = (df['Anatomical Site'] == 'P').mean() * 100
            st.metric("Pulmonary TB", f"{pulmonary_pct:.1f}%")
        with col2:
            bact_confirmed = (df['Bacteriologic Status'] == 'Bacteriologically-confirmed TB').mean() * 100
            st.metric("Bacteriologically Confirmed", f"{bact_confirmed:.1f}%")
        with col3:
            # Cured or Treatment Completed
            success_outcomes = df['Outcome/Status'].isin(['CURED', 'TREATMENT COMPLETED']).mean() * 100
            st.metric("Treatment Success Rate", f"{success_outcomes:.1f}%")
        with col4:
            # Died
            mortality = (df['Outcome/Status'] == 'DIED').mean() * 100
            st.metric("Mortality Rate", f"{mortality:.1f}%")
        
        # Row 3: Additional insights
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            new_cases = (df['Registration Group'] == 'NEW').mean() * 100
            st.metric("New Cases", f"{new_cases:.1f}%")
        with col2:
            valid_ages = df[(df['Age'] >= 0) & (df['Age'] <= 120)]['Age']
            median_age = valid_ages.median()
            st.metric("Median Age", f"{median_age:.0f} years")
        with col3:
            female_pct = (df['Sex'] == 'F').mean() * 100
            st.metric("Female %", f"{female_pct:.1f}%")
        with col4:
            # Lost to follow-up
            ltfu = (df['Outcome/Status'] == 'LOST TO FF-UP').mean() * 100
            st.metric("Lost to Follow-up", f"{ltfu:.1f}%")
        
        st.divider()
        
        # Section: Temporal Analysis
        st.markdown("## 📅 Temporal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Cases by Year")
            year_counts = df['Year'].value_counts().sort_index()
            fig_year = px.line(
                x=year_counts.index.astype(int),
                y=year_counts.values,
                labels={'x': 'Year', 'y': 'Number of Cases'},
                markers=True
            )
            fig_year.update_traces(line_color='#1f77b4', line_width=2, marker_size=8)
            fig_year.update_layout(height=350)
            st.plotly_chart(fig_year, use_container_width=True)
        
        with col2:
            st.markdown("### Cases by Year and Sex")
            df_trend = df.copy()
            df_trend['Sex_Label'] = df_trend['Sex'].map(sex_map)
            trend_data = df_trend.groupby(['Year', 'Sex_Label']).size().reset_index(name='Count')
            fig_trend = px.line(
                trend_data,
                x='Year',
                y='Count',
                color='Sex_Label',
                markers=True,
                labels={'Count': 'Number of Cases', 'Sex_Label': 'Sex'}
            )
            fig_trend.update_layout(height=350)
            st.plotly_chart(fig_trend, use_container_width=True)
        
        st.divider()
        
        # Section: Demographic Analysis
        st.markdown("## 👥 Demographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Sex Distribution")
            sex_counts = df['Sex'].map(sex_map).value_counts()
            fig_sex = px.pie(
                values=sex_counts.values,
                names=sex_counts.index,
                color_discrete_sequence=['#ff7f0e', '#1f77b4']
            )
            fig_sex.update_layout(height=350)
            st.plotly_chart(fig_sex, use_container_width=True)
        
        with col2:
            st.markdown("### Age Distribution")
            valid_ages = df[(df['Age'] >= 0) & (df['Age'] <= 120)]['Age']
            fig_age = px.histogram(
                valid_ages,
                nbins=30,
                labels={'value': 'Age', 'count': 'Frequency'},
                color_discrete_sequence=['#2ca02c']
            )
            fig_age.update_layout(height=350, xaxis_title="Age", yaxis_title="Count", showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)
        
        # Age-Sex Pyramid
        st.markdown("### Age-Sex Pyramid")
        df_valid = df[(df['Age'] >= 0) & (df['Age'] <= 120)].copy()
        df_valid['Age_Group'] = pd.cut(
            df_valid['Age'],
            bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 120],
            labels=['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+']
        )
        df_valid['Sex_Label'] = df_valid['Sex'].map(sex_map)
        
        pyramid_data = df_valid.groupby(['Age_Group', 'Sex_Label']).size().reset_index(name='Count')
        male_data = pyramid_data[pyramid_data['Sex_Label'] == 'Male'].copy()
        female_data = pyramid_data[pyramid_data['Sex_Label'] == 'Female'].copy()
        male_data['Count'] = -male_data['Count']  # Negative for left side
        
        fig_pyramid = go.Figure()
        fig_pyramid.add_trace(go.Bar(
            y=male_data['Age_Group'], x=male_data['Count'],
            orientation='h', name='Male', marker_color='steelblue'
        ))
        fig_pyramid.add_trace(go.Bar(
            y=female_data['Age_Group'], x=female_data['Count'],
            orientation='h', name='Female', marker_color='salmon'
        ))
        fig_pyramid.update_layout(
            barmode='overlay', height=400,
            xaxis=dict(title='Number of Cases', tickvals=[-2000, -1000, 0, 1000, 2000],
                      ticktext=['2000', '1000', '0', '1000', '2000']),
            yaxis=dict(title='Age Group')
        )
        st.plotly_chart(fig_pyramid, use_container_width=True)
        
        st.divider()
        
        # Section: Clinical Characteristics
        st.markdown("## 🏥 Clinical Characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Anatomical Site")
            site_counts = df['Anatomical Site'].map(anatomical_site_map).value_counts()
            fig_site = px.pie(
                values=site_counts.values,
                names=site_counts.index,
                color_discrete_sequence=['#d62728', '#17becf']
            )
            fig_site.update_layout(height=350)
            st.plotly_chart(fig_site, use_container_width=True)
        
        with col2:
            st.markdown("### Bacteriologic Status")
            bact_counts = df['Bacteriologic Status'].map(bacteriologic_map).value_counts()
            fig_bact = px.pie(
                values=bact_counts.values,
                names=bact_counts.index,
                color_discrete_sequence=['#9467bd', '#8c564b']
            )
            fig_bact.update_layout(height=350)
            st.plotly_chart(fig_bact, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Registration Group")
            reg_counts = df['Registration Group'].map(registration_map).value_counts()
            fig_reg = px.bar(
                x=reg_counts.values,
                y=reg_counts.index,
                orientation='h',
                labels={'x': 'Count', 'y': 'Registration Group'},
                color_discrete_sequence=['#e377c2']
            )
            fig_reg.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_reg, use_container_width=True)
        
        with col2:
            st.markdown("### Source of Patient")
            source_counts = df['Source of Patient'].map(source_map).value_counts()
            fig_source = px.bar(
                x=source_counts.values,
                y=source_counts.index,
                orientation='h',
                labels={'x': 'Count', 'y': 'Source'},
                color_discrete_sequence=['#7f7f7f']
            )
            fig_source.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_source, use_container_width=True)
        
        st.divider()
        
        # Section: Treatment Outcomes
        st.markdown("## 📋 Treatment Outcomes")
        
        

        st.markdown("### Outcome/Status Distribution")
        outcome_counts = df['Outcome/Status'].map(outcome_map).value_counts()
        fig_outcome = px.bar(
            x=outcome_counts.values,
            y=outcome_counts.index,
            orientation='h',
            labels={'x': 'Count', 'y': 'Outcome'},
            color_discrete_sequence=['#17becf']
        )
        fig_outcome.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_outcome, use_container_width=True)
    

        st.markdown("### Outcome Trend by Year")
        df_outcome = df.copy()
        df_outcome['Outcome_Label'] = df_outcome['Outcome/Status'].map(outcome_map)
        outcome_year = df_outcome.groupby(['Year', 'Outcome_Label']).size().reset_index(name='Count')
        # Filter top outcomes for cleaner visualization
        top_outcomes = outcome_counts.head(5).index.tolist()
        outcome_year_filtered = outcome_year[outcome_year['Outcome_Label'].isin(top_outcomes)]
        
        # Add total cases per year
        total_by_year = df.groupby('Year').size().reset_index(name='Count')
        total_by_year['Outcome_Label'] = 'Total Cases'
        
        # Combine outcome trends with total cases
        outcome_year_with_total = pd.concat([outcome_year_filtered, total_by_year], ignore_index=True)
        
        fig_outcome_trend = px.line(
            outcome_year_with_total,
            x='Year',
            y='Count',
            color='Outcome_Label',
            markers=True,
            labels={'Count': 'Number of Cases', 'Outcome_Label': 'Outcome'}
        )
        # Make Total Cases line stand out (dashed, thicker)
        fig_outcome_trend.for_each_trace(
            lambda trace: trace.update(line=dict(dash='dash', width=3)) if trace.name == 'Total Cases' else ()
        )
        fig_outcome_trend.update_layout(height=400)
        st.plotly_chart(fig_outcome_trend, use_container_width=True)
        
        st.divider()
        
        # Data Preview Section
        st.markdown("## 📄 Data Preview")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            year_filter = st.multiselect("Filter by Year", options=sorted(df['Year'].dropna().unique().astype(int)), default=[])
        with col2:
            sex_filter = st.multiselect("Filter by Sex", options=['Male', 'Female'], default=[])
        with col3:
            outcome_filter = st.multiselect("Filter by Outcome", options=list(outcome_map.values()), default=[])
        
        # Apply filters
        filtered_df = df.copy()
        if year_filter:
            filtered_df = filtered_df[filtered_df['Year'].isin(year_filter)]
        if sex_filter:
            sex_values = ['F' if s == 'Female' else 'M' for s in sex_filter]
            filtered_df = filtered_df[filtered_df['Sex'].isin(sex_values)]
        if outcome_filter:
            outcome_values = [k for k, v in outcome_map.items() if v in outcome_filter]
            filtered_df = filtered_df[filtered_df['Outcome/Status'].isin(outcome_values)]
        
        st.write(f"Showing {len(filtered_df):,} records")
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure the file '2015-2025-consolidated-clean.csv' exists in the dataset/non-temporal/ folder.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
