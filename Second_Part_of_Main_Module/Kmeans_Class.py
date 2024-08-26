import sys, os
# add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Second_Part_of_Main_Module.Pearson_corr_and_LinReg_Task2 import *
import warnings
import matplotlib.cm as cm


k = 4
#if __name__ == "__main__" or __name__ == "gui_for_prediction_ifChurn.py":
class Kmeans():
    def __init__(self, cluster_num, centroids, if_loopStopCond_is_numOfIters=False,
                 if_loopStopCond_is_error_between_centroids=False, iter_num=5, needed_error=0.1):
        self.cluster_num = cluster_num
        self.centroids = centroids
        self.iter_num = iter_num
        self.needed_error = needed_error
        self.error_real = 43230
        self.if_loopStopCond_is_numOfIters = if_loopStopCond_is_numOfIters
        self.if_loopStopCond_is_error_between_centroids = if_loopStopCond_is_error_between_centroids

        self.error_real = [float('inf')]


    #@classmethod
    def datapoint_centroids_distance(cls,X,centroids) -> list:  # X=datapoints (1dimen.,2dimen.,...ndimen.) #euclid distance between a datapoint and
        # all of centroids
        '''Calculating the Euclidean Distance between each datapoint and all clusters '''
        distances = [np.sqrt(np.sum((np.array(X) - np.array(centroid)) ** 2)) for centroid in centroids]
        return distances

    #     def old_new_c_distance(self,old,new) -> list: #distance between old and new centroids
    #         res_list = [np.sqrt(sum((new[i][dim_num] - old[i][dim_num])**2 + (new[i][dim_num+1] - old[i][dim_num+1])**2 \
    #         for dim_num in range(len(old[i])-1))) for i in range(len(old))]
    #         return res_list
    def old_new_c_distance(self, old, new):
        res_list = [np.sqrt(sum((new[i][dim_num] - old[i][dim_num]) ** 2
                                for dim_num in range(len(old[i])))) for i in range(len(old))]
        return res_list

    def make_clusters(self, X):  # fit_data
        # initializing centroids

        # 1. assign a datapoint to a cluster (using euclidean distance)
        # 2. calculate the mean in each cluster,for all datapoints in this cluster
        # (np.mean(cluster[i]) for i in range(len(centroids))) - it will be the new centroid (for each cluster)
        # 3. reassign clusters (reassign datapoints to new clusters (with new, just-calculated centroids))
        # 4. repeat until a condition (iter < num_of_iters or new centroids are not getting updated) is satisfied

        iteration = 0
        clusters = [[] for _ in range(self.cluster_num)]
        # if we have 3cs, it would be: [ [], [], [] ]
        next_centroids = self.centroids
        next_clusters = [[] for _ in range(self.cluster_num)]


        # iter_num is default (if 0 and 0)
        cond_is_iterNum_andNotError = self.if_loopStopCond_is_numOfIters == 1 and self.if_loopStopCond_is_error_between_centroids == 0
        while_condition_iterNum = iteration < self.iter_num

        cond_is_Error_andNotIternum = self.if_loopStopCond_is_error_between_centroids == 1 and self.if_loopStopCond_is_numOfIters ==0
        while_condition_error = max(self.error_real) > self.needed_error

        cond_is_iterNum_and_Error = self.if_loopStopCond_is_numOfIters ==1 and self.if_loopStopCond_is_error_between_centroids==1
        while_condition_iterNum_and_Error = iteration < self.iter_num and max(self.error_real) > self.needed_error

        default_condition_iterNum = self.if_loopStopCond_is_numOfIters == 0 and self.iter_num == 0
        while_condition_default = while_condition_iterNum # the same as iter_num

        list_of_conditions_if = [cond_is_iterNum_andNotError,cond_is_Error_andNotIternum, cond_is_iterNum_and_Error,default_condition_iterNum]
        list_of_conditions_if_while = [while_condition_iterNum,while_condition_error,while_condition_iterNum_and_Error,while_condition_default]

        true_index = list_of_conditions_if.index(1)
        if 1 in list_of_conditions_if:
            while_cond_user_dependent = list_of_conditions_if_while[true_index]


        #while iteration < self.iter_num or max(self.error_real) > self.needed_error:  # error
        while while_cond_user_dependent:  #
            #             clusters = [[] for _ in range(self.cluster_num)]
            #             # if we have 3cs, it would be: [ [], [], [] ]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.centroids = next_centroids
                clusters = next_clusters

                # 1. assign a datapoint to a cluster (using euclidean distance): #each datapoint?
                for g in range(X.shape[0]):
                    distances_b_datapoint_and_clusters = self.datapoint_centroids_distance(X[g],self.centroids)  # got a list like [6,2,8] where each number is
                    # distance between this datapoint and each cluster
                    indx_min_distance_cluster = np.argmin(distances_b_datapoint_and_clusters)
                    clusters[indx_min_distance_cluster].append(X[g])  # add a datapoint to a cluster with the smallest distance between this point and centroid of a cluster


                # 2. calculate the mean in each cluster,for all datapoints in this cluster
                # clusters: [[1,2],[2,3],[3,4]]
                # next_centroids = [list(np.mean(cl, axis=0)) for cl in clusters]
                next_centroids = [np.mean(cl, axis=0).tolist() for cl in clusters]

                # 3. reassign clusters:
                next_clusters = [[] for _ in range(self.cluster_num)]  # len(self.cluster_num)
                # 3. reassign clusters (next datapoint to new clusters)
                for ng in range(X.shape[0]):
                    distances_b_datapoint_and_new_clusters = self.datapoint_centroids_distance(X[ng], next_centroids)
                    indx_min_distance_new_cluster = np.argmin(distances_b_datapoint_and_new_clusters)
                    next_clusters[indx_min_distance_new_cluster].append(X[ng])

                self.error_real = self.old_new_c_distance(self.centroids, next_centroids);print('\n')  # old,new
                print(f"Old centroids: {self.centroids}, \nnew centroids: {next_centroids}")
                print(f"Error: {self.error_real} \n\n")
                colors = ['r', 'g', 'b', 'y']
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for each_cl in range(len(clusters)):
                    if clusters[each_cl]:  # check if the cluster is not empty
                        cluster_points = np.array(clusters[each_cl])
                        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], s=20,
                                   color=colors[each_cl], label=f'Cluster {each_cl}')
                centroids_array = np.array(next_centroids)
                ax.scatter(centroids_array[:, 0], centroids_array[:, 1], centroids_array[:, 2], s=100, color='brown',
                           marker='X', label='Centroids')
                ax.set_title(f"Iteration {iteration}");ax.legend();plt.show()


                if cond_is_iterNum_andNotError: #1,0
                    if iteration >= self.iter_num:
                        print(f"----parameters: iter_num=1, error=0....This is the last iteration...iter_num:{iteration}, error:{self.error_real}")
                        break

                if cond_is_Error_andNotIternum: #0,1
                    if max(self.error_real) <= self.needed_error:
                        print(f"----parameters: iter_num=0, error=1....This is the last iteration...iter_num:{iteration}, error:{self.error_real}")
                        break

                if cond_is_iterNum_and_Error: #1,1
                    if iteration >= self.iter_num and max(self.error_real) <= self.needed_error:
                        print(f"----parameters: iter_num=1, error=1....This is the last iteration...iter_num:{iteration}, error:{self.error_real}")
                        break

                if default_condition_iterNum: #0,0
                    if iteration >= self.iter_num:
                        print(f"----parameters: iter_num=0, error=0....This is the last iteration...iter_num:{iteration}, error:{self.error_real}")
                        break
                iteration += 1
        return next_centroids


    # class method for labeling clusters (churn = 1 or 0)
    def label_clusters(self,all_centroids) -> list:
        cmean = []
        for h in range(len(all_centroids)):
            ctr_mean = np.mean(all_centroids[h])
            cmean.append(ctr_mean)
        list_of_labels_forEachCluster = ['yes' if each_cmean >= np.mean(cmean) else 'no' for each_cmean in cmean]
        return list_of_labels_forEachCluster #['yes', 'no', 'no', 'no'] #for example

    #the parameter would be one datapoint like X=[201, 45, 0, 1] #where each number is x1,x2,x3,x4
    def predict_if_churn(self,datapoint,all_centroids,X) -> str:
        dp_c_distances = self.datapoint_centroids_distance(datapoint,all_centroids)

        centroids_ = self.make_clusters(X)
        label_yes_or_no = self.label_clusters(centroids_)
        min_distance = min(dp_c_distances)
        ind_min_dist = dp_c_distances.index(min_distance)
        churn = label_yes_or_no[ind_min_dist]

        message_if_churn = ''
        if churn == 'yes': message_if_churn = "Prediction: Client has stopped using this company's services"
        else: message_if_churn = "Prediction: Client has not stopped using this company's services"

        return message_if_churn



x1_total_day_mins = an_cpy['total_day_minutes']
x2_total_day_charge = an_cpy['total_day_charge']
x3_num_of_customer_service_calls = an_cpy['number_customer_service_calls'] #0-9
x4_Has_intern_plan = an_cpy['Has_intern_plan'] #0,1
features_to_concat = [x1_total_day_mins,x2_total_day_charge,x3_num_of_customer_service_calls,x4_Has_intern_plan]
X = pd.concat(features_to_concat, axis=1)


#if __name__ == "__main__" or __name__ == "gui_for_prediction_ifChurn.py":
# # initializing centroids
centroids = []
for _ in range(k):
    centroid = []
    for i in range(X.shape[1]):
        min_X = X.iloc[:, i].min()
        max_X = X.iloc[:, i].max()
        if i==2 or i==3:
            centroid.append(np.random.randint(min_X,max_X+1))
        else:
            centroid.append(np.random.uniform(min_X, max_X))
    centroids.append(centroid)
print(centroids)

print(f"----First centroids:",'\n')
for c in centroids:
    print(c,'\n')


# #if __name__ == "__main__":
s=X.to_numpy()
X = list(s)
Xnp = np.array(X)
#Kmeans(k, centroids, 1, 0, 6, 0.1)
kmeans = Kmeans(k, centroids, 1, 0, 5, 0.1) #last_centroids
print(f"------------------------------------------------------------------------------------------------------------Data Clustering Process starting now------------------------------------------------------------------------------------------------------------")
print(kmeans,'\n')
kmeans.make_clusters(Xnp)
last_centroids = kmeans.make_clusters(Xnp)
print(f"------------------------------------------------------------------------------------------------------------Data Clustering Process finished------------------------------------------------------------------------------------------------------------\n")
print(f"Last Centroids (Kmeans algorithm result): {last_centroids}\n")
#yes_no = Kmeans.label_clusters(kmeans, last_centroids)
yes_no = kmeans.label_clusters(last_centroids)
ir = [*range(len(last_centroids))]
for i in range(len(ir)):
    print(f"____Churn Result for Cluster {i+1}: {yes_no[i]}\n")

res = kmeans.predict_if_churn([201.43, 54.12, 8, 1], last_centroids,Xnp) # 210, 53.8, 7, 0 - churn==1
#188, 41.83, 6, 1

#40, 2.4, 9, 0
print(res)

#__all__ = [centroids,last_centroids, Xnp]
