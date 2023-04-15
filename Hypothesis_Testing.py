import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from Generating_Sample import df_psa, df_ad

# evaluate whether the average amount of ads seen by user is equal between control and treatment group
# start by doing F-test to check if we should use T-test with equal or unequal variance
control = np.array(df_psa['total ads'])
treatment = np.array(df_ad['total ads'])
f = np.var(control, ddof=1) / np.var(treatment, ddof=1)
dfn = control.size - 1
dfd = treatment.size - 1
p = 1 - stats.f.cdf(f, dfn, dfd)
if p < 0.05:
    var_equality = False
else:
    var_equality = True
print(f"p-value of F-test is {p}, variance equality is {var_equality}")

# continue doing two-sided T-test with equal variance to evaluate whether the average amount of ads seen by control
# and treatment group is equal
result = stats.ttest_ind(a=df_ad['total ads'],
                         b=df_psa['total ads'],
                         equal_var=var_equality,
                         alternative="two-sided")
if result.pvalue < 0.5:
    print("Null hypothesis rejected, average amount of ads seen by treatment and control group is unequal")
else:
    print("Fail to reject null hypothesis, average amount of ads seen by treatment and control group is equal")

# calculate conversion rate of control and treatment group
psa_sample_conv = len(df_psa[df_psa['converted'] == True])
ad_sample_conv = len(df_ad[df_ad['converted'] == True])
n_psa = len(df_psa)
n_ad = len(df_ad)
psa_sample_conv_rate = psa_sample_conv / n_psa
ad_sample_conv_rate = ad_sample_conv / n_ad
print(f'conversion rate of psa is {psa_sample_conv_rate}, ad is {ad_sample_conv_rate}')

# conduct two sample z-test for proportion
count_conv = [ad_sample_conv, psa_sample_conv]
count_obs = [n_ad, n_psa]
diff = 0.01
alternative_ops = 'larger'
z_stat, p_value = proportions_ztest(count=count_conv,
                                    nobs=count_obs,
                                    value=diff,
                                    alternative=alternative_ops)
if p_value < 0.05:
    print("Reject Null Hypothesis, the difference between control and treatment group is larger than 1%")
else:
    print("Fail to reject Null Hypothesis, the difference between control and treatment group is not larger than 1%")

# calculate the confidence interval
confidence_interval = confint_proportions_2indep(count1=ad_sample_conv, nobs1=n_ad,
                                                 count2=psa_sample_conv, nobs2=n_psa,
                                                 compare='diff', alpha=0.05)
print(f'With 95% confidence level, we know that the difference in conversion rate of treatment and control group is '
      f'between {confidence_interval[0]} to {confidence_interval[1]}')
