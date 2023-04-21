import cv2
import numpy as np
import matplotlib.pyplot as plt

scale_factor = 0.2
threshold_for_matching = 0.7

first_image_t = cv2.imread('C:\Masters\Spring 2023\ENPM673_Perception\Perception\Project2\problem_2_images\image_1.jpg')
resized_first_image_t = cv2.resize(first_image_t, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

second_image_t = cv2.imread('C:\Masters\Spring 2023\ENPM673_Perception\Perception\Project2\problem_2_images\image_2.jpg')
resized_second_image_t = cv2.resize(second_image_t, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

third_image_t = cv2.imread('C:\Masters\Spring 2023\ENPM673_Perception\Perception\Project2\problem_2_images\image_3.jpg')
resized_third_image_t = cv2.resize(third_image_t, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

fourth_image_t = cv2.imread('C:\Masters\Spring 2023\ENPM673_Perception\Perception\Project2\problem_2_images\image_4.jpg')
resized_fourth_image_t = cv2.resize(fourth_image_t, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

def matched(first_image,second_image):
    resized_first_image= cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    resized_second_image= cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)
    

    sift = cv2.SIFT_create()

    feature_p_1, desription_p_1 = sift.detectAndCompute(resized_first_image, None)
    feature_p_2, desription_p_2 = sift.detectAndCompute(resized_second_image, None)

    matching_features_1 = cv2.FlannBasedMatcher()
    features_matched_1 = matching_features_1.knnMatch(desription_p_1, desription_p_2, k=2)

    best_matched_features_1 = []
    for i,j in features_matched_1:
        if i.distance < threshold_for_matching*j.distance:
            best_matched_features_1.append(i)

     
    matched_images_1 = cv2.drawMatches(resized_first_image, feature_p_1, resized_second_image, feature_p_2, best_matched_features_1, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return matched_images_1

def panaroma(first_image,second_image):

    resized_first_image= cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    resized_second_image= cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)
    

    sift = cv2.SIFT_create()

    feature_p_1, desription_p_1 = sift.detectAndCompute(resized_first_image, None)
    feature_p_2, desription_p_2 = sift.detectAndCompute(resized_second_image, None)

    matching_features_1 = cv2.FlannBasedMatcher()
    features_matched_1 = matching_features_1.knnMatch(desription_p_1, desription_p_2, k=2)

    best_matched_features_1 = []
    for i,j in features_matched_1:
        if i.distance < threshold_for_matching*j.distance:
            best_matched_features_1.append(i)

     
    source_point_1 = np.float32([feature_p_1[m.queryIdx].pt for m in best_matched_features_1 ]).reshape(-1,1,2)
    destination_points_1 = np.float32([feature_p_2[m.trainIdx].pt for m in best_matched_features_1 ]).reshape(-1,1,2)

    threshold_for_ransac = 10
    itr_for_ransac = 1000


    best_homogr_mat = None
    min_inlier_ransac = 0

    for i in range(itr_for_ransac):
        index_at_rand = np.random.choice(len(source_point_1), 4, replace= False)
        trial_sample_1 = source_point_1[index_at_rand]
        trial_sample_2 = destination_points_1[index_at_rand]

        Matrix_A_list = []
        for i in range(len(trial_sample_2)):
            x_source, y_source = trial_sample_2[i][0][0], trial_sample_2[i][0][1]
            x_hat, y_hat = trial_sample_1[i][0][0], trial_sample_1[i][0][1]
            Matrix_A_list.append(np.array([
            [x_source, y_source, 1, 0, 0, 0, -x_hat*x_source, -x_hat*y_source, -x_hat],
            [0, 0, 0, x_source, y_source, 1, -y_hat*x_source, -y_hat*y_source, -y_hat]
        ]))
            matrix_A = np.empty([0, Matrix_A_list[0].shape[1]])

        for i in Matrix_A_list:
         matrix_A = np.append(matrix_A, i, axis=0)

   
        first_eig, second_eig = np.linalg.eig(matrix_A.T @ matrix_A)

    
        homogr = second_eig[:, np.argmin(first_eig)]
   
        homo_mat = homogr.reshape((3, 3))

        homogr_source = np.concatenate(
        (destination_points_1, np.ones((len(destination_points_1), 1, 1), dtype=np.float32)), axis=2)

        updated_homogr_pts = np.matmul(homogr_source, homo_mat.T)

        updated_pts = updated_homogr_pts[:, :, :2] / updated_homogr_pts[:, :, 2:]

       
        delta_d = np.linalg.norm(source_point_1 - updated_pts, axis = 2)
        no_of_inlier = np.sum(delta_d<threshold_for_ransac)

        if no_of_inlier > min_inlier_ransac:
            min_inlier_ransac = no_of_inlier
            best_homogr_mat = homo_mat
    
      
    image_stitched = cv2.warpPerspective(second_image, best_homogr_mat, ((first_image.shape[1] + second_image.shape[1]), first_image.shape[0]))
    image_stitched[0:first_image.shape[0], 0:first_image.shape[1]] = first_image


    return image_stitched


# To match and visualise the features between consecutive images:
stitched_12 = matched(resized_first_image_t,resized_second_image_t)

cv2.imshow('Features mathced and visualized in first and second image', stitched_12)
cv2.waitKey(0)
cv2.destroyAllWindows()


stitched_23 = matched(resized_second_image_t,resized_third_image_t)

cv2.imshow('Features mathced and visualized  in second and third image', stitched_23)
cv2.waitKey(0)
cv2.destroyAllWindows()

stitched_34 = matched(resized_third_image_t, resized_fourth_image_t)

cv2.imshow('Features mathced and visualized  in third and fourth image', stitched_34)
cv2.waitKey(0)
cv2.destroyAllWindows()

#TO combine these four images together:

panaroma_1 = panaroma(resized_third_image_t,resized_fourth_image_t)

panaroma_2 = panaroma(resized_second_image_t, panaroma_1)

panaroma_3 = panaroma(resized_first_image_t, panaroma_2)

cv2.imshow('Final Panaroma of four images', panaroma_3)
cv2.waitKey(0)
cv2.destroyAllWindows()

