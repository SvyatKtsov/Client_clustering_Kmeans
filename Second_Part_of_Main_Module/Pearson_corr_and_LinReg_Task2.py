import sys, os
# add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy import stats
from First_Part_of_Main_Module.Data_exploration import *
#from First_Part_of_Main_Module import Data_exploration
#from scipy import *
#import statistics as stats

if __name__ == "__main__":

    r_pearson_x1, r_pearson_x2, r_pearson_x3, r_pearson_x4, r_pearson_x5 = (0, 0, 0, 0, 0,)
    s1_x1, s2_x1, s1_x2, s2_x2, s1_x3, s2_x3, s1_x4, s2_x4, s1_x5, s2_x5 = (0, 0, 0, 0, 0,) * 2
    # s1_x1, s2_x1, s1_x2, s2_x2 = (0,0,0,0,)
    s2_first_p, s2_second_p = (0, 0,)
    for i in range(len(y_churn_yes)):
        s1_x1 += (x_number_vmail_messages[i] - np.mean(x_number_vmail_messages)) * (y_churn_yes[i] - np.mean(y_churn_yes))
    for i in range(len(y_churn_yes)):
        s2_first_p += (x_number_vmail_messages[i] - np.mean(x_number_vmail_messages)) ** 2
    for i in range(len(y_churn_yes)):
        s2_second_p += (y_churn_yes[i] - np.mean(y_churn_yes)) ** 2

    s2_x1 = s2_first_p * s2_second_p
    s2_x1 = s2_x1 ** 0.5
    r_pearson_x1 = s1_x1 / s2_x1
    print(f"Коеф.Пірсона, \"К-ість відправлених клієнтом голосових електронних адрес \
    \"-\"Чи перестав клієнт користуватися послугами компанії\"{r_pearson_x1}")

    ######################################
    # оновлюємо значення двох сум (суми чисельника та знаменника) і обчислюємо коеф.Пірса для x2:
    # оновлюємо значення сум чисельника й знаменника для коеф.Пірса№3
    s2_first_p, s2_second_p = (0, 0,)
    for j in range(len(y_churn_yes)):
        s1_x2 += (x_total_intl_calls[j] - np.mean(x_total_intl_calls)) * (y_churn_yes[j] - np.mean(y_churn_yes))
    for j in range(len(y_churn_yes)):
        s2_first_p += (x_total_intl_calls[j] - np.mean(x_total_intl_calls)) ** 2
    for j in range(len(y_churn_yes)):
        s2_second_p += (y_churn_yes[j] - np.mean(y_churn_yes)) ** 2
    # оновлюємо значення сум чисельника й знаменника для коеф.Пірса№2

    s2_x2 = s2_first_p * s2_second_p
    s2_x2 = s2_x2 ** 0.5
    r_pearson_x2 = s1_x2 / s2_x2
    print(f"Коеф.Пірсона, \"К-ість (міжнародних) телефонних дзвінків\"-\"Чи перестав клієнт користуватися послугами компанії\
    \"{r_pearson_x2}")

    ###################################################
    s2_first_p, s2_second_p = (0, 0,)
    for j in range(len(y_churn_yes)):
        s1_x3 += (x_number_customer_service_calls[j] - np.mean(x_number_customer_service_calls)) * (
                    y_churn_yes[j] - np.mean(y_churn_yes))
    for j in range(len(y_churn_yes)):
        s2_first_p += (x_number_customer_service_calls[j] - np.mean(x_number_customer_service_calls)) ** 2
    for j in range(len(y_churn_yes)):
        s2_second_p += (y_churn_yes[j] - np.mean(y_churn_yes)) ** 2

    s2_x3 = s2_first_p * s2_second_p
    s2_x3 = s2_x3 ** 0.5
    r_pearson_x3 = s1_x3 / s2_x3
    print(
        f"Коеф.Пірсона, \"К-ість дзвінків у сервіс підтримки\"-\"Чи перестав клієнт користуватися послугами компанії\"{r_pearson_x3}")

    ############################
    # WV
    s2_first_p, s2_second_p = (0, 0,)
    for j in range(len(y_churn_yes)):
        s1_x4 += (x_state_WestVirginia[j] - np.mean(x_state_WestVirginia)) * (y_churn_yes[j] - np.mean(y_churn_yes))
    for j in range(len(y_churn_yes)):
        s2_first_p += (x_state_WestVirginia[j] - np.mean(x_state_WestVirginia)) ** 2
    for j in range(len(y_churn_yes)):
        s2_second_p += (y_churn_yes[j] - np.mean(y_churn_yes)) ** 2

    s2_x4 = s2_first_p * s2_second_p
    s2_x4 = s2_x4 ** 0.5
    r_pearson_x4 = s1_x4 / s2_x4
    print(
        f"Коеф.Пірсона, \"Чи клієнт зареєстрований у ЗВ\"-\"Чи перестав клієнт користуватися послугами компанії\"{r_pearson_x4}")

    ##############################
    # MN
    s1_x1 = 0
    s1_x2 = 0
    s2_first_p, s2_second_p = (0, 0,)
    for j in range(len(y_churn_yes)):
        s1_x5 += (x_state_Minnesota[j] - np.mean(x_state_Minnesota)) * (y_churn_yes[j] - np.mean(y_churn_yes))
    for j in range(len(y_churn_yes)):
        s2_first_p += (x_state_Minnesota[j] - np.mean(x_state_Minnesota)) ** 2
    for j in range(len(y_churn_yes)):
        s2_second_p += (y_churn_yes[j] - np.mean(y_churn_yes)) ** 2

    s2_x5 = s2_first_p * s2_second_p
    s2_x5 = s2_x5 ** 0.5
    r_pearson_x5 = s1_x5 / s2_x5
    print(
        f"Коеф.Пірсона, \"Чи клієнт зареєстрований у МН\"-\"Чи перестав клієнт користуватися послугами компанії\"{r_pearson_x5}")


    ############################################################################# checking if the results, gotten from the
                                                                                        #implementation, are correct, using scipy
    from scipy import *
    rp_x1 = stats.pearsonr(x_number_vmail_messages, y_churn_yes)
    rp_x2 = stats.pearsonr(x_total_intl_calls, y_churn_yes)
    rp_x3 = stats.pearsonr(x_number_customer_service_calls, y_churn_yes)
    rp_x4 = stats.pearsonr(x_state_WestVirginia, y_churn_yes)
    rp_x5 = stats.pearsonr(x_state_Minnesota, y_churn_yes)

    print(rp_x4, rp_x5)
    for each_chosen_col in all_needed_cols_forLoopPC:
        print(f"Pearson_coeff for {each_chosen_col} and y_churn_yes is:{stats.pearsonr(df_copy[each_chosen_col], y_churn_yes)[0]}",'\n')

#y_churn_yes
features_for_clustering = [ch_col for ch_col in all_needed_cols_forLoopPC if stats.pearsonr(df[ch_col],y_churn_yes)[0]>0.2]
if __name__ == '__main__':
    print('==='*40)
    print("Features with the highest correlation rate:{}".format(features_for_clustering))


    cols = list(df.columns)[6:20]
    # stats.pearsonr(df_copy['total_day_minutes'],df_copy['total_day_calls'])
    for k in cols[1:]:
        # print(stats.pearsonr(df_copy['number_vmail_messages'],df_copy[k]))
        print(stats.pearsonr(df_copy[k], df_copy['number_vmail_messages']))

    print('\n');
    print(stats.pearsonr(df_copy['total_eve_minutes'], df_copy['total_eve_charge']))
    linreg_twocols = pd.DataFrame()
    linreg_twocols = pd.concat([df_copy['total_eve_minutes'], df_copy['total_eve_charge']], axis=1)
    linreg_twocols

    # чи розірвав клієнт контракт із компанією з надання телекомунікаційних послуг, якщо так - 1
    #del i
    plt.scatter(df_copy['total_eve_minutes'], df_copy['total_eve_charge'],marker='*')


##################################################################################################
### Linear Regression
an_cpy = df.copy()
international_plan_oneHotEn, voice_mail_plan_oneHotEn = pd.get_dummies(an_cpy['international_plan']), \
pd.get_dummies(an_cpy['voice_mail_plan'])
an_cpy = pd.concat([an_cpy, international_plan_oneHotEn],axis=1)
an_cpy.drop(columns=['no'],axis=1)
an_cpy = an_cpy.rename(columns={'yes':'Has_intern_plan'}) #rename 'yes for international_plan'
an_cpy = pd.concat([an_cpy, voice_mail_plan_oneHotEn],axis=1)
an_cpy.drop(columns=['no'],axis=1)
an_cpy = an_cpy.rename(columns={'yes':'Has_voiceMail_plan'}) #rename 'yes for voice_mail_plan'
new_an_cpy = an_cpy.drop(columns=['no','Has_voiceMail_plan']) .copy()

voice_mail_plan_oneHotEn = pd.get_dummies(new_an_cpy['voice_mail_plan'])
new_an_cpy = pd.concat([new_an_cpy, voice_mail_plan_oneHotEn],axis=1)
new_an_cpy.drop(columns=['no'],axis=1)
new_an_cpy = new_an_cpy.rename(columns={'yes':'Has_voiceMail_plan'}) #rename 'yes for voice_mail_plan'

if __name__ == '__main__':
    print(stats.pearsonr(new_an_cpy['Has_intern_plan'], y_churn_yes))
    stats.pearsonr(new_an_cpy['Has_voiceMail_plan'], y_churn_yes)
    #plot
    plt.scatter(df_copy['total_eve_minutes'], df_copy['total_eve_charge'],c='g',marker='*',)


