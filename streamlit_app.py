def xai_insights_page():
    st.markdown("<h2 style='color: #33aaff;'>XAI Insights</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #666;'>
    Explainable AI (XAI) helps understand how the model makes predictions using SHAP values.
    </p>
    """, unsafe_allow_html=True)

    # Try to import SHAP with error handling
    try:
        import shap
        shap_available = True
    except ImportError:
        shap_available = False
        st.warning("SHAP library not installed - showing simplified explanations")
        show_fallback_xai()
        return

    # Load model with error handling
    with st.spinner("Loading model for explanation..."):
        model = load_model()
        if model is None:
            st.error("Model not loaded - cannot generate explanations")
            show_fallback_xai()
            return

    # Create sample data that matches model expectations
    sample_data = pd.DataFrame({
        'weekly_visits': [3],
        'total_dependents_3_months': [2],
        'pickup_count_last_30_days': [4],
        'pickup_count_last_14_days': [2],
        'Holidays': [0],
        'pickup_week': [25],
        'time_since_first_visit': [90]
    })

    # SHAP Explanation Section
    try:
        with st.spinner("Generating SHAP explanations..."):
            # Initialize explainer with robust settings
            explainer = shap.TreeExplainer(
                model,
                feature_perturbation="interventional",
                model_output="probability"
            )

            # Calculate SHAP values
            shap_values = explainer.shap_values(sample_data, check_additivity=False)

            # Ensure we have valid SHAP values
            if shap_values is None or len(shap_values) == 0:
                raise ValueError("SHAP values computation returned empty results")

            # Global Feature Importance Plot
            st.markdown("### Global Feature Importance")
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    sample_data,
                    plot_type="bar",
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()
            except Exception as e:
                st.error(f"Could not generate global importance plot: {str(e)}")

            # Individual Prediction Explanation
            st.markdown("### Individual Prediction Explanation")
            try:
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
            except Exception as e:
                st.error(f"Could not generate individual explanation: {str(e)}")

            # Interpretation Guide
            st.markdown("""
            **How to Interpret These Results**:
            
            - **Global Importance**: Shows which features most influence predictions overall
            - **Individual Explanation**: Shows how each feature contributes to this specific prediction
            - Longer bars indicate stronger influence on the prediction
            - Blue bars indicate features increasing return probability
            - Red bars indicate features decreasing return probability
            """)

    except Exception as e:
        st.error(f"SHAP explanation failed: {str(e)}")
        st.info("""
        Common solutions:
        1. Ensure your model was trained with the same features shown above
        2. Verify all feature values are within expected ranges
        3. Try different sample data values
        """)
        show_fallback_xai()

def show_fallback_xai():
    """Show simplified XAI when SHAP is not available or fails"""
    st.markdown("### Simplified Feature Importance")
    
    # Create example visualization
    features = [
        'weekly_visits',
        'total_dependents_3_months',
        'pickup_count_last_30_days',
        'pickup_count_last_14_days',
        'Holidays',
        'pickup_week',
        'time_since_first_visit'
    ]
    importance = [0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importance, y=features, palette="viridis", ax=ax)
    ax.set_title("Feature Importance (Example)")
    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("Features")
    st.pyplot(fig)
    
    st.markdown("""
    **Key Insights**:
    - Weekly visits is the most important predictor
    - Number of dependents is the second most important
    - Recent pickup activity strongly influences predictions
    - Holidays and pickup week have smaller but still significant effects
    """)
