#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import numpy as np
import scipy
import scipy.stats as st


# In[2]:


booking = pd.read_csv('AB_test_data.csv')


# ## 1. Conduct an A/B test to determine whether Alternative B improved conversion rates (site users book the property) over alternative A.

# In[3]:


booking.head()


# In[4]:


booking_A = booking[booking['Variant'] == "A"]
booking_B = booking[booking['Variant'] == "B"]


# In[5]:


booking_B


# In[6]:


#calculate 'True' percentage in A group: p = 0.149616

p = booking_A[(booking_A['purchase_TF'] == True)].id.count()/booking_A.id.count()
print(p)


# In[7]:


#calculate 'True' percentage in B group: p_head = 0.1766

p_head = booking_B[(booking_B['purchase_TF'] == True)].id.count()/booking_B.id.count()
print(p_head)


# In[8]:


#null hypo: p head = p
#alternative hypo: p head > p


# In[9]:


#Calculate z-score = 5.349273094732516

n = booking_B.shape[0]

numerator = p_head-p
denominator = math.sqrt((p*(1-p))/n)

numerator / denominator


# In[10]:


# calculate z critical value = 1.6448536269514722

scipy.stats.norm.ppf(1-.05)


#z-score > z critical value, reject null hypothesis.
#There is enough evidence (alpha = 5%) to support the claim that B improves conversion rates.


# ## 2. Calculate the optimal sample size for a 95% confidence rate and test with 80% power. Conduct the test 10 times using samples of the optimal size. Report results.

# In[11]:


#confidence rate = 95% --> probability of type I error: 5%
#power = 80% --> probability of type II error: 20%


# In[12]:


#delta: difference between the two means. We use 1%.

p_bar = (p + p_head)/2

#difference of conversion rate 
optimal_sample = (st.norm.ppf(0.975)*math.sqrt(2*p_bar*(1-p_bar)) + st.norm.ppf(0.8)*math.sqrt(p*(1-p)+p_head*(1-p_head)))**2 /((p_head-p)**2)
optimal_sample = int(optimal_sample)

print("The optimal size should be:", optimal_sample)


# In[13]:


test1a = booking_A.sample(n = optimal_sample) 
test2a = booking_A.sample(n = optimal_sample) 
test3a = booking_A.sample(n = optimal_sample) 
test4a = booking_A.sample(n = optimal_sample)
test5a = booking_A.sample(n = optimal_sample)
test6a = booking_A.sample(n = optimal_sample)
test7a = booking_A.sample(n = optimal_sample)
test8a = booking_A.sample(n = optimal_sample)
test9a = booking_A.sample(n = optimal_sample)
test10a = booking_A.sample(n = optimal_sample)

test1b = booking_B.sample(n = optimal_sample) 
test2b = booking_B.sample(n = optimal_sample) 
test3b = booking_B.sample(n = optimal_sample) 
test4b = booking_B.sample(n = optimal_sample)
test5b = booking_B.sample(n = optimal_sample)
test6b = booking_B.sample(n = optimal_sample)
test7b = booking_B.sample(n = optimal_sample)
test8b = booking_B.sample(n = optimal_sample)
test9b = booking_B.sample(n = optimal_sample)
test10b = booking_B.sample(n = optimal_sample)

a_sample_list = [test1a, test2a, test3a, test4a, test5a, test6a, test7a, test8a, test9a, test10a]
b_sample_list = [test1b, test2b, test3b, test4b, test5b, test6b, test7b, test8b, test9b, test10b]


# In[14]:


a_sample_list[5]


# In[15]:


a_mean_list = []
b_mean_list = []

for i in a_sample_list:
    a_mean_list.append(float(np.mean(i["purchase_TF"])))
    
for i in b_sample_list:
    b_mean_list.append(float(np.mean(i["purchase_TF"])))

mean_diff_list = []
for i in range(10):
    mean_diff_list.append(b_mean_list[i] - a_mean_list[i])


# In[16]:


mean_diff_list


# In[17]:


sigma_list = []

for i in range(10):
    p = a_mean_list[i]
    sigma_list.append(math.sqrt((p*(1-p))/n))


# In[18]:


sigma_list


# In[19]:


z_score_list = []
for i in range(10):
    z_score_list.append(mean_diff_list[i]/sigma_list[i])


# In[20]:


z_score_list


# In[21]:


significance_list = []
for i in z_score_list:
    if i > scipy.stats.norm.ppf(1-.05):
        significance_list.append(True)
    else:
        significance_list.append(False)


# In[22]:


significance_list


# In[23]:


for i in significance_list:
    if i == True:
        print('WOW!!significant improvement!')
    else:
        print('oops')


# ## 3. Conduct a sequential test for the 10 samples. For any of the samples, were you able to stop the test prior to using the full sample? What was the average number of iterations required to stop the test?

# In[24]:


upper_bound = np.log(1/(1-0.95))
lower_bound = np.log(1-0.8)
rounds_ran = []
reason = []

for i in range(10):
    print(i)
    log_gamma = 0
    rounds = 0
    
    while (log_gamma > lower_bound) & (log_gamma < upper_bound):
        if rounds < optimal_sample:
            if b_sample_list[i]['purchase_TF'].values[rounds]:
                log_gamma = log_gamma + math.log(p_head / p)
            else:
                log_gamma = log_gamma + math.log((1-p_head) / (1-p))
            rounds += 1
        else:
            reason.append('Did not stop early')
            break

    rounds_ran.append(rounds)
    if log_gamma < lower_bound:
        reason.append('Lower bound')
    elif log_gamma > upper_bound:
        reason.append('Higher bound')


# In[25]:


rounds_ran


# In[26]:


reason


# In[35]:


from statistics import mean
mean(rounds_ran)


# In[37]:


#(mean(rounds_ran[0:1]+rounds_ran[2:]))


# In[29]:


#We're able to stop the test 9 out of 10 times, with an average of stopping at ~1278 rounds/iterations.
#Or if not including the one time not stopping early and going to 2317 rounds, then an average of ~1094 rounds/iterations

#Note, that this varies greatly when rerunning with different seeds/randomizations.
#This run is a pretty balanced one in terms of having just one non-stopping eraly,
#and 2 lower bounds as well as 7 upper bound reasons for stopping


# In[ ]:





# In[30]:


print(upper_bound)
print(lower_bound)


# In[ ]:




