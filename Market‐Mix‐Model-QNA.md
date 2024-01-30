# Q1. Explain NRMSE along with its equation, and also tell how NRMSE is better than RMSE?
Ans: In most of the regression analysis tasks, we often use RMSE(Root Mean Square Error) as an evaluation metric to check the model’s performance. But what if we had to compare two models trained on two different Target variables? Can we use Root Mean Square Error as an evaluation metric, considering both Target variables would have gone through different kinds of data transformation techniques (like log transformation, square root, standardized, etc.)  to get the best-fit line or best model to capture the pattern in data? Using RMSE to compare these two models would be like comparing apples with oranges.
 
That's where NRMSE comes to the rescue. As the name explains, NRMSE is normalized RMSE. You can normalize the RMSE in the following ways depending on the business use case- 
 
![image](https://github.com/Praveen76/Machine-Learning-Interview-preparation/assets/26660076/a21c24bd-f8c3-4747-b494-9fa2e5bfa05b)

If the response variables have few extreme values, the interquartile range should be preferred as it is less sensitive to outliers 
 
Downside: 
Since NRMSE is a normalized evaluation metric, it’ll lose the units associated with the response variable.

The Normalized Root Mean Squared Error (NRMSE) is a normalized version of the Root Mean Squared Error (RMSE), and it is used as an evaluation metric in regression analysis. Both NRMSE and RMSE are measures of the differences between predicted and actual values, but NRMSE has the advantage of being scale-independent, making it potentially more interpretable and comparable across different datasets.

Here's a brief explanation of RMSE and NRMSE:

1. **Root Mean Squared Error (RMSE):**
   - RMSE is a measure of the average magnitude of the errors between predicted and observed values in a regression analysis.
   - It is calculated as the square root of the average of the squared differences between predicted and actual values.
   - The formula for RMSE is often expressed as: 
  $$\ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \$$

  Here, $n$ is the number of observations, $y_i$ is the actual value, and $\\hat{y}_i\$ is the predicted value for observation $i$.

2. **Normalized Root Mean Squared Error (NRMSE):**
   - NRMSE is a normalized version of RMSE, which means it is scaled by the range of the target variable.
   - The formula for NRMSE is often expressed as: 

$$\ NRMSE = \frac{RMSE}{\text{max}(y) - \text{min}(y)} \$$

Here, 

- $RMSE$ stands for Root Mean Square Error, and $\text{max}(y) - \text{min}(y)\$ represents the range of the observed values in the true data set.

- **$\ \text{max}(y) - \text{min}(y)\ $:** This represents the range of the observed values in the true data set. It helps normalize the RMSE by the scale of the data, making the NRMSE a dimensionless measure.


Advantages of NRMSE over RMSE:

1. **Scale-Independence:**
   - NRMSE is normalized by the range of the target variable, making it scale-independent. This allows for better comparability between different datasets, as it is not affected by the scale of the values.

2. **Interpretability:**
   - NRMSE provides a normalized measure that can be interpreted as the proportion of the RMSE relative to the range of the target variable. This can be useful when comparing models on datasets with different scales.

3. **Consistency Across Datasets:**
   - NRMSE can be more consistent across datasets with varying scales, making it easier to assess model performance in diverse scenarios.

It's important to note that while NRMSE has certain advantages, the choice between RMSE and NRMSE depends on the specific requirements of your analysis and the nature of your dataset. Always consider the characteristics of your data and the goals of your modeling efforts when selecting an evaluation metric.

# Q2. Please explain The Mean Absolute Percentage Error (MAPE.LIFT) along with its equation in a Market-Mix-Model. 
Ans: 

Def 1: [The Mean Absolute Percentage Error (MAPE.LIFT) is a metric that measures the accuracy of a Market-Mix-Model (MMM) in predicting the incremental sales lift due to marketing activities](https://www.statisticshowto.com/mean-absolute-percentage-error-mape/).

Def 2: The MAPE.LIFT formula is a metric that measures the difference between the causal effect and the predicted effect of marketing variables on sales in a market-mix-model. It is defined as:

$$
\text{MAPE.LIFT} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i - \hat{y}_i^{\text{base}}} \right| \times 100
$$

where $y_i$ is the actual sales, $\hat{y}_i$ is the predicted sales with marketing variables, and $\hat{y}_i^{\text{base}}$ is the predicted sales without marketing variables for the $i$-th observation. The lower the MAPE.LIFT, the better the model fits the data.

The MAPE.LIFT metric is similar to the standard MAPE metric, but it only considers the incremental sales lift, which is the difference between the actual sales and the baseline sales. The baseline sales are the sales that would have occurred without any marketing activities.

The MAPE.LIFT metric can be used to evaluate the performance of different MMMs and to compare them with other forecasting methods. A lower MAPE.LIFT value indicates a higher accuracy of the MMM. However, the MAPE.LIFT metric has some limitations, such as:

- It can be skewed by outliers or extreme values in the data.
- It can be undefined or infinite if the actual incremental sales lift is zero or very close to zero.
- It can be biased by the scale or magnitude of the data.

Therefore, the MAPE.LIFT metric should be used with caution and in conjunction with other metrics, such as the mean absolute error (MAE), the root mean squared error (RMSE), or the R-squared.


# Q3: What is DECOMP.RSSD in a Market-Mix-Model, and how is it helpful?
Ans:
DECOMP.RSSD stands for **decomposition root sum of squared distance**. It is a metric invented by the Robyn team at Facebook (now Meta) to measure how much the model agrees with the current budget allocation across different marketing channels¹². It is calculated as the square root of the sum of the squared differences between the effect share and the spend share of each channel.


$$
\text{DECOMP.RSSD} = \sqrt{\sum_{i=1}^{n} (\text{Effect Share}_i - \text{Spend Share}_i)^2}
$$

where $n$ is the number of channels, $\text{Effect Share}_i$ is the effect share of channel $i$, and $\text{Spend Share}_i$ is the spend share of channel $i$ .

Effect share is the proportion of sales that the model attributes to a channel, while spend share is the proportion of budget that is spent on a channel. A low DECOMP.RSSD means that the model's results are consistent with the current spending strategy, while a high DECOMP.RSSD means that the model suggests a different allocation of budget to optimize the return on advertising spend (ROAS).

DECOMP.RSSD is helpful because it incorporates business logic and plausibility into the model selection process. It helps to avoid models that have high accuracy but unrealistic parameter values, such as negative or very small effects for some channels. It also helps to align the model's recommendations with the business goals and expectations, and to facilitate the adoption of the model by the decision-makers. However, DECOMP.RSSD is also controversial, because it may bias the model towards not revealing the true effectiveness of each channel, and may prevent the discovery of potential opportunities for improving the marketing mix.

# Q4: What is return on advertising spend (RoAS) in a Market Mix Model and how is it helpful?
Ans: Return on advertising spend (RoAS) is a marketing metric that estimates the amount of revenue earned per dollar allocated to advertising. It is similar to the return on investment (ROI) metric, but specific to the context of analyzing advertising spend.

ROAS is helpful because it measures the cost-effectiveness of marketing campaigns and related spending. It helps businesses to evaluate which advertising channels are doing well and how they can improve their advertising efforts in the future to increase sales or revenue. It also helps to compare the performance of different campaigns, platforms, or ads, and to optimize the budget allocation across them.

The formula for calculating ROAS is:

$$
\text{RoAS} = \frac{\text{Conversion Revenue}}{\text{Advertising Spend}}
$$

where conversion revenue is the amount of revenue brought in from the ad campaigns, and advertising spend is the amount of capital spent on ad campaigns and adjacent activities. A ROAS greater than 1 means that the ad campaign is generating more revenue than its cost, while a ROAS less than 1 means that the ad campaign is losing money. However, ROAS alone does not indicate the profitability of a business, as there are other expenses that have to be deducted before determining the net profit margin.

# Q 5: What are the SHAP Values?
Ans: SHAP (SHapley Additive exPlanations) values are a concept borrowed from cooperative game theory and applied to machine learning models for the purpose of explaining the output of a model for a specific instance or prediction. They were introduced by Lundberg and Lee in their 2017 paper "A Unified Approach to Interpreting Model Predictions."

Here's a brief explanation of SHAP values:

1. **Definition:**
   - SHAP values provide a way to fairly distribute the contribution of each feature to the prediction made by a machine learning model.
   - They are based on the Shapley values concept from cooperative game theory, which allocates a value to each player in a cooperative game based on their marginal contributions.

2. **Interpretation:**
   - In the context of machine learning, each feature of an input contributes to the model's output. SHAP values aim to assign a contribution value to each feature such that the sum of contributions equals the difference between the model's prediction for a specific instance and the average prediction for all instances.
   - Positive SHAP values indicate a positive contribution to the prediction, while negative values indicate a negative contribution.

3. **Calculation:**
   - SHAP values are computed by considering all possible feature combinations and measuring the change in the model's prediction when a particular feature is added to or removed from the combination. This process is computationally expensive, but efficient algorithms, such as TreeSHAP for tree-based models, have been developed to approximate SHAP values more quickly.

4. **Use Cases:**
   - SHAP values provide a way to interpret complex machine learning models, such as ensemble models and deep neural networks.
   - They are often used for feature importance analysis, helping to understand which features have the most significant impact on a model's predictions.

5. **Visualization:**
   - SHAP values can be visualized in various ways, such as summary plots, force plots, and waterfall charts. These visualizations help in understanding the contribution of each feature to a specific prediction.

In the provided code snippet, SHAP values are used to analyze the impact of different media channels on the model's predictions, specifically in the context of advertising spend. You can calculate the absolute sum of SHAP values for each media channel and then incorporate this information into the calculation of spend effect shares.

# Q3: What will be the equation for spend effect share, if you're using SHAP Values to understand impact of each media channel?
Ans:
Certainly! Based on the provided code, it looks like you are calculating the spend effect share using Shap values and original spend data. The spend effect share is calculated as the ratio of the absolute sum of Shap values for each media channel to the total sum of Shap values, and the ratio of the sum of original spends for each media channel to the total sum of original spends. Here's the mathematical equation:

Let:
- $\( R_i \)$: Absolute sum of Shap values for media channel \( i \)
- $\( S_i \)$: Sum of original spends for media channel \( i \)
- $\( \text{Total Shap Values} \)$: Total sum of absolute Shap values for all media channels
- $\( \text{Total Original Spends} \)$: Total sum of original spends for all media channels

The spend effect share $(\( SES_i \))$ for each media channel $\( i \)$ can be calculated as follows:

$$\[ SES_i = \frac{R_i}{\text{Total Shap Values}} \times \frac{S_i}{\text{Total Original Spends}} \]$$

# Q4: What does Absolute sum of Shap values for media channel means?
Ans: The absolute sum of SHAP (SHapley Additive exPlanations) values for a media channel represents the total impact of that particular media channel on a model's prediction, regardless of the direction (positive or negative) of the individual SHAP values.

Here's a breakdown of the concept:

1. **SHAP Values for a Feature:**
   - In the context of machine learning models, SHAP values represent the contribution of each feature to the model's prediction for a specific instance.
   - For a given prediction, each feature (in this case, a media channel) has an associated SHAP value, indicating the impact of that feature on the model's output.

2. **Absolute Sum of SHAP Values for a Media Channel:**
   - The absolute sum is obtained by taking the sum of the magnitudes (absolute values) of the individual SHAP values for a specific media channel.
   - Mathematically, it is calculated as follows:
     \[ \text{Absolute Sum for Media Channel} = \sum_{i=1}^{N} |SHAP\_value_i| \]
   - This total represents the overall impact of the media channel, irrespective of whether the impact is positive or negative.

3. **Interpretation:**
   - If the absolute sum for a media channel is high, it indicates that the presence of that media channel has a substantial influence on the model's prediction across various instances.
   - High absolute sum suggests that the corresponding media channel plays a significant role in determining the output of the model.

4. **Application in the Code:**
   - In the provided code, the absolute sum of SHAP values for each media channel is calculated using the `abs().sum(axis=0)` operation on the SHAP values dataframe for those media channels.
   - This absolute sum is then used to compute the spend effect share, which provides insights into the proportional contribution of each media channel to the overall impact on the model's predictions.

In summary, the absolute sum of SHAP values for a media channel is a way of quantifying the total impact of that channel on the model's predictions, considering both positive and negative contributions.


