import numpy as np
import scipy
from pr3_utils import *
import functions_library as FL
from functions_library import *

if __name__ == '__main__':
    
        # Load the measurements
        filename = "../data/03.npz"
        t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
        len_t=(t.shape[1])
        tao=np.diff(t[0,:])
        M=features.shape[1]

        print(features.shape)
        IMU_pose_H=np.arange(4*4*len_t).reshape((4,4,len_t)).astype('float64')

        #-------------------initialize------------------------
        IMU_pose_H[:,:,0]=np.array([[1,0,0,0],
                                      [0,1,0,0],
                                      [0,0,1,0],
                                      [0,0,0,1]])
        mu=np.zeros((3,M)).astype('float64')
        mu_sigma=np.zeros((3,3,M))
        fsu=K[0,0]
        fsv=K[1,1]
        cu=K[0,2]
        cv=K[1,2]
        b=0.6
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
        print('oTi=\n',cam_T_imu)
        
        #-------------------initialize mu@mu_sigma----------------------
        i=0
        t=0
        while i<M:
                if sum(features[:,i,t])>-4:
                        mu[:,i]=landmark_init(features[:,i,t],IMU_pose_H[:,:,t],imu_T_cam_f,fsu,b,K)
                i=i+1
        i=0
        while i<M:
                mu_sigma[:,:,i]=W*np.eye(3)
                i=i+1

        #------------------------------------------------------        
        # (a) IMU Localization via EKF Prediction
        Twist=np.vstack((linear_velocity,angular_velocity)).T
        Twist_hat=axangle2twist(Twist)
        #T=axangle2pose(Twist)
        IMU_pose=np.array([[[1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]]]).astype('float64')
        i=0
        while i<len_t-1:
                T=scipy.linalg.expm(Twist_hat[i]*tao[i])
                IMU_pose_H[:,:,i+1]=IMU_pose_H[:,:,i]@T
                #IMU_pose=np.vstack((IMU_pose,[IMU_pose[i]@T]))
                #IMU_pose_H[:,:,i+1]=IMU_pose[i]@T
                i=i+1




        #----------------------------------------------------------
	# (b) Landmark Mapping via EKF Update
        loop=len_t-1
        #loop=100
        t=0
        while t<loop:
                #print('t=',t)
                i=0
                while i<M:
                        H=np.zeros((4,3)).astype('float64')
                        if(sum(features[:,i,t+1])>-4):
                                if(np.array_equal(mu[:,i],[0,0,0])):
                                        #print('i=',i)
                                        mu[:,i]=FL.landmark_init(features[:,i,t+1],IMU_pose_H[:,:,t+1],imu_T_cam_f,fsu,b,K)
                                else:
                                        mu[:,i],mu_sigma[:,:,i]=FL.landmark_update(features[:,i,t+1],mu[:,i],
                                                                mu_sigma[:,:,i],IMU_pose_H[:,:,t+1],
                                                                cam_T_imu,Ks,P_trans,V)
                                        '''
                                        z=features[:,i,t+1]
                                        q=cam_T_imu@np.linalg.inv(IMU_pose_H[:,:,t+1])@(np.hstack((mu[:,i],[1])))                                        
                                        z_predict=Ks@FL.pi_norm(q)
                                        A=cam_T_imu@np.linalg.inv(IMU_pose_H[:,:,t+1])@P_trans
                                        dq=FL.pi_Jacobian(q)
                                        H=Ks@dq@A

                                        Kal=mu_sigma[:,:,i]@(H.T)@np.linalg.inv((H@mu_sigma[:,:,i]@(H.T)+V*np.eye(4)))

                                        mu[:,i]=mu[:,i]+Kal@(z-z_predict)
                                        mu_sigma[:,:,i]=(np.eye(3)-Kal@H)@mu_sigma[:,:,i]
                                        '''


                                
                        i=i+1
                         
                #print(t+1)
                if t%100==0:
                        print('-----t=',t+1)

                t=t+1







	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
        
        #print(T.shape)
        #print(IMU_pose.shape)
        #print('End Pose:',IMU_pose[-1,:,:])
        #print(IMU_pose_H[:,:,-1])
	
        visualize_trajectory_2d(IMU_pose_H[:,:,0:loop+1],mu,path_name="Unknown",show_ori=True)

