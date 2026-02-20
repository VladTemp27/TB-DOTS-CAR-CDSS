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

tab1, tab2 = st.tabs(["Temporal", "Non Temporal"])

with tab1:
    st.write("This is where the temporal data will be displayed.")

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
