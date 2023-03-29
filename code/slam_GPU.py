import numpy as np
import scipy
from pr3_utils import *
import functions_library as FL
from functions_library import *
import torch
if __name__ == '__main__':
        a=torch.ones([3,3]).cuda()
        print(a)
        # Load the measurements
        filename = "../data/03.npz"
        t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
        len_t=(t.shape[1])
        tao=np.diff(t[0,:])
        M=features.shape[1]
        
        print(features.shape)
        pose=np.arange(4*4*len_t).reshape((4,4,len_t)).astype('float64')

        #-------------------initialize------------------------
        pose[:,:,0]=np.array([[1,0,0,0],
                                      [0,1,0,0],
                                      [0,0,1,0],
                                      [0,0,0,1]])
        m=np.zeros((3,M)).astype('float64')
        m_sigma=np.zeros((3,3,M))
        fsu=K[0,0]
        fsv=K[1,1]
        cu=K[0,2]
        cv=K[1,2]
        Ks=np.array([[fsu,0,cu,0],
                    [0,fsv,cv,0],
                    [fsu,0,cu,-fsu*b],
                    [0,fsv,cv,0]]).astype('float64')

        imu_T_cam_f=np.array([[1,0,0],
                              [0,-1,0],
                              [0,0,-1]])@imu_T_cam[0:3,0:3]
        imu_T_cam_f=np.hstack((imu_T_cam_f,imu_T_cam[0:3,3].reshape((3,1))))
        imu_T_cam_f=np.vstack((imu_T_cam_f,np.array([0,0,0,1]).reshape(1,4)))
        cam_T_imu=np.linalg.inv(imu_T_cam_f)
        P_trans=np.array([[1,0,0],
                          [0,1,0],
                          [0,0,1],
                          [0,0,0]]).astype('float64')
        W=0.01
        V=1
        cov_pose=0.01
        pose_sigma=np.zeros((6,6)).astype('float64')
        W_pose=cov_pose*np.eye(6)
        Twist=np.vstack((linear_velocity,angular_velocity)).T
        Twist_hat=axangle2twist(Twist)
        #-------------------initialize mu@mu_sigma----------------------
        i=0
        t=0
        while i<M:
                if sum(features[:,i,t])>-4:
                        m[:,i]=landmark_init(features[:,i,t],pose[:,:,t],imu_T_cam_f,fsu,b,K)
                i=i+1
        i=0
        while i<M:
                m_sigma[:,:,i]=W*np.eye(3)
                i=i+1

        #------------------------------------------------------
        #(c) Visual-Inertial SLAM
        scale=10
        loop=len_t-1
        #loop=3
        t=0
        while t<loop:
                print('t=',t)
                #(c1) pose prediction
                pose_predict=pose[:,:,t]@scipy.linalg.expm(Twist_hat[t]*tao[t])  #pose is miu in lecture
                pose_sigma_predict=pose_sigma_predict_EKF(tao[t],Twist[t],pose_sigma,W_pose)
                #(c2) landmark EKF& update H_pose
                H_pose=np.zeros((4*(int(M/scale)+1),6)).astype('float64')
                z_observe=np.zeros((4*(int(M/scale)+1))).astype('float64')
                z_predict=np.zeros((4*(int(M/scale)+1))).astype('float64')
                error_twist=np.zeros((6))
                error_sigma_pose=np.zeros((6,6))
                i=0
                while i<M:
                        if(sum(features[:,i,t+1])>-4):
                                if(np.array_equal(m[:,i],[0,0,0])):                                     
                                        m[:,i]=FL.landmark_init(features[:,i,t+1],pose_predict,imu_T_cam_f,fsu,b,K)
                                else:
                                        m[:,i],m_sigma[:,:,i]=FL.landmark_update(features[:,i,t+1],m[:,i],
                                                                m_sigma[:,:,i],pose_predict,
                                                                cam_T_imu,Ks,P_trans,V)
                                if(i%scale==0):
                                        #print('i=',i)
                                        j=int(i/scale)
                                        #print('j=',j)
                                        z_observe[4*j:4*j+4]=features[:,i,t+1]
                                        H_pose[4*j:4*j+4,:],z_predict[4*j:4*j+4]=get_H_z_pose(Ks,cam_T_imu,pose_predict,m[:,i])
                        i=i+1

                #print('start pose update')
                #(c3) pose EKF update
                mat_V=(torch.eye(H_pose.shape[0])*V).cuda()
                H_poseG=torch.tensor(H_pose).cuda()
                pose_sigma_predictG=torch.tensor(pose_sigma_predict).cuda()
                
                mat_A=H_poseG@pose_sigma_predictG@(H_poseG.T)+mat_V
                Kal_pose=(pose_sigma_predictG@(H_poseG.T)@torch.linalg.inv(mat_A)).cpu().numpy()
                pose[:,:,t+1]=pose_predict@axangle2pose(Kal_pose@(z_observe-z_predict))
                pose_sigma=(np.eye(6)-Kal_pose@H_pose)@pose_sigma_predict

                #print('update done!')
                t=t+1
        mfile="../code/mid_data/m_dataset_03.npy"
        posefile="../code/mid_data/pose_dataset_03.npy"
        np.save(mfile,m)
        np.save(posefile,pose[:,:,0:loop+1])
        visualize_trajectory_2d(pose[:,:,0:loop+1],m,path_name="Unknown",show_ori=True)


	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
        
        #print(T.shape)
        #print(IMU_pose.shape)
        #print('End Pose:',IMU_pose[-1,:,:])
        #print(IMU_pose_H[:,:,-1])


