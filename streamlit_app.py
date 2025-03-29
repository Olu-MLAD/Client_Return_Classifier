def xai_insights_page():
    st.markdown("<h2 style='color: #33aaff;'>XAI Insights</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #666;'>
    Explainable AI (XAI) helps understand how the model makes predictions using SHAP values.
    </p>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading prediction model..."):
        model = load_model()
    if model is None:
        st.error("Failed to load model - cannot generate explanations")
        show_fallback_xai()
        return

    # Create sample data that matches model's training format
    X = pd.DataFrame({
        'weekly_visits': [3, 1, 4],
        'total_dependents_3_months': [2, 1, 3],
        'pickup_count_last_30_days': [4, 2, 5],
        'pickup_count_last_14_days': [2, 1, 3],
        'Holidays': [0, 0, 1],
        'pickup_week': [25, 10, 50],
        'time_since_first_visit': [90, 30, 180]
    })

    try:
        # Compute SHAP values with correct settings
        with st.spinner("Computing SHAP explanations..."):
            # Use interventional perturbation which supports probability output
            explainer = shap.TreeExplainer(
                model,
                feature_perturbation="interventional",
                model_output="probability"
            )
            
            # Calculate SHAP values with additivity check disabled
            shap_values = explainer.shap_values(X, check_additivity=False)

            # SHAP Summary Plot (Bar Chart)
            st.markdown("### Feature Importance (SHAP Values)")
            fig, ax = plt.subplots(figsize=(12, 6))
            shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
            plt.title("Which Features Most Influence Predictions?")
            st.pyplot(fig)
            plt.close()

            # Detailed SHAP summary plot
            st.markdown("### How Feature Values Affect Predictions")
            fig, ax = plt.subplots(figsize=(12, 6))
            shap.summary_plot(shap_values[1], X, show=False)
            plt.title("Feature Value Impact on Predictions")
            st.pyplot(fig)
            plt.close()

            st.markdown("""
            **Interpreting the Results**:
            - **Feature Importance**: Shows which factors most influence predictions
            - **Value Impact**: Shows how feature values affect outcomes
            - Right of center = increases return probability
            - Left of center = decreases return probability
            - Color shows feature value (red=high, blue=low)
            """)

    except Exception as e:
        st.error(f"Detailed explanation failed: {str(e)}")
        show_fallback_xai()
