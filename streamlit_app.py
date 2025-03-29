def xai_insights_page():
    st.markdown("<h2 style='color: #33aaff;'>XAI Insights</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #666;'>
    Explainable AI (XAI) helps understand how the model makes predictions using SHAP values.
    </p>
    """, unsafe_allow_html=True)
    
    try:
        import shap
        st.success("SHAP library successfully imported!")
        
        with st.spinner("Loading model explainability..."):
            model = load_model()
            if model is None:
                st.error("Model not loaded - cannot generate explanations")
                return
            
            try:
                # Create sample data that matches the model's expected input shape
                sample_data = pd.DataFrame({
                    'weekly_visits': [3],
                    'total_dependents_3_months': [2],
                    'pickup_count_last_30_days': [4],
                    'pickup_count_last_14_days': [2],
                    'Holidays': [0],
                    'pickup_week': [25],
                    'time_since_first_visit': [90]
                })
                
                # Create SHAP explainer with robust settings
                explainer = shap.TreeExplainer(
                    model,
                    feature_perturbation="interventional",
                    model_output="probability"
                )
                
                # Calculate SHAP values with additivity check disabled
                shap_values = explainer.shap_values(
                    sample_data,
                    check_additivity=False
                )
                
                # Plot SHAP summary
                st.markdown("### Global Feature Importance")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    sample_data,
                    plot_type="bar",
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig)
                
                # Individual prediction explanation
                st.markdown("### Individual Prediction Explanation")
                st.markdown("How each feature contributes to a single prediction:")
                
                plt.figure()
                shap.force_plot(
                    explainer.expected_value[1],
                    shap_values[1][0],
                    sample_data.iloc[0],
                    matplotlib=True,
                    show=False
                )
                st.pyplot(plt.gcf(), bbox_inches='tight')
                plt.clf()
                
                # Feature dependence plots
                st.markdown("### Feature Dependence")
                st.markdown("How predictions change with feature values:")
                
                for feature in sample_data.columns:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    shap.dependence_plot(
                        feature,
                        shap_values[1],
                        sample_data,
                        display_features=sample_data,
                        show=False,
                        ax=ax
                    )
                    st.pyplot(fig)
                
                st.markdown("""
                **Interpreting SHAP Values**:
                - **Global Importance**: Shows which features most influence predictions overall
                - **Individual Explanation**: Shows how each feature contributes to a specific prediction
                - **Dependence Plots**: Show how changing a feature affects the prediction
                - Blue values increase likelihood of returning
                - Red values decrease likelihood of returning
                """)
                
            except Exception as e:
                st.error(f"SHAP explanation failed: {str(e)}")
                st.info("""
                Common solutions:
                1. Ensure your model was trained with the same features shown above
                2. Check that all feature values are within expected ranges
                3. Try different sample data values
                """)
                show_fallback_xai()
                
    except ImportError:
        st.warning("SHAP library not installed - showing simplified explanations")
        show_fallback_xai()
