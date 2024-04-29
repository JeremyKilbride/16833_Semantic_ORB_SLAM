import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_output', default='../build/output_data.txt')
parser.add_argument('dataset_name')

args = parser.parse_args()
name=args.dataset_name
src_path_output = args.path_to_output
most_recent_pair=[0,0]
num_inliers=[]
sequentail_inliers=[]
sequentail_pairs=[]
angles=[]
angles_second_first=[]
angles_first_second=[]
raw_scores_first_first=[]
raw_scores_first_second=[]
raw_scores_second_second=[]
raw_scores_second_first=[]
current_label=""
out_file=open(src_path_output,"r")
for line in out_file:
    new_line=line.rstrip("\n")
    list_line=new_line.split(" ")
    if (list_line[0]=="path:") or(list_line[0]=="num_images:"):
        pass
    if list_line[0]=="N:":
        num_inliers.append(int(list_line[1]))
        if (most_recent_pair[1]-most_recent_pair[0])==1 :
            sequentail_inliers.append(int(list_line[1]))
    if list_line[0]=="P:":
        pair=list_line[1]
        img_indxs=pair.split("/")
        img_indxs=[int(img_indxs[0]),int(img_indxs[1])]
        rel_angle=(img_indxs[1]-img_indxs[0])*2
        angles.append(rel_angle)
        most_recent_pair=img_indxs
        if (most_recent_pair[1]-most_recent_pair[0])==1 :
            sequentail_pairs.append(pair)
    if list_line[0]=="L:":
        current_label=list_line[1]
    if list_line[0]=="S:":
        if current_label=="forklift1":
            raw_scores_first_first.append(float(list_line[1]))
        elif current_label=="forklift2":
            raw_scores_second_second.append(float(list_line[1]))
    if list_line[0]=="S2:":
        if current_label=="forklift1":
            raw_scores_first_second.append(float(list_line[1]))
            rel_angle_1=most_recent_pair[1]-most_recent_pair[0]
            angles_first_second.append(rel_angle_1)
        elif current_label=="forklift2":
            raw_scores_second_first.append(float(list_line[1]))
            rel_angle_2=most_recent_pair[1]-most_recent_pair[0]
            angles_second_first.append(rel_angle_2)




plt.figure()
plt.scatter(angles,num_inliers,c='r',s=5)
plt.ylim((0,300))
plt.title("Relative angle vs Number of inliers ")
plt.xlabel("Relative angle (degrees)")
plt.ylabel("Number of inliers")
plt.savefig('rel_angle_v_inliers'+name+'.png')


plt.figure()
plt.plot(sequentail_pairs,sequentail_inliers)
plt.ylim((0,300))
plt.title("number of inliers in sequential image pairs")
plt.ylabel("Number of inliers")
plt.xlabel("image pair")
plt.savefig("sequential_frame_inliers"+name+".png")


plt.figure()
plt.scatter(angles,raw_scores_first_first,s=8)
plt.scatter(angles_first_second,raw_scores_first_second,s=8)
plt.scatter(angles,raw_scores_second_second,s=8)
plt.scatter(angles_second_first,raw_scores_second_first,s=8)
plt.legend(["forklift1-forklift1","forklift1-forklift2","forklift2-forklift2","forklift2-forklift1"])
plt.yscale("log")
plt.title("Relative angle vs Raw L1 Scores")
plt.xlabel("Relative angle (degrees)")
plt.ylabel("Score")
plt.savefig('rel_angle_v_score'+name+'.png')
