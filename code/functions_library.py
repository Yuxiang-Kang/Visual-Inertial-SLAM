import numpy as np
import pr3_utils as util
import scipy
def define_map(xmin,ymin,xmax,ymax, resolution):
    M={}
    M = {}
    M['res']   = resolution #meter
    M['xmin']  = xmin  #meter
    M['ymin']  = ymin
    M['xmax']  = xmax
    M['ymax']  = ymax
    M['sizex']  = int(np.ceil((M['xmax'] - M['xmin']) / M['res'] + 1)) #cells
    M['sizey']  = int(np.ceil((M['ymax'] - M['ymin']) / M['res'] + 1))
    M['map'] = np.zeros((M['sizex'],M['sizey']),dtype=float) #DATA TYPE: char or int8

    return M

def pi_norm(q):
    pi_q=1/q[2]*q
    return pi_q

def pi_Jacobian(q):
    #q is a (4) array
    dpdq=1/q[2]*np.array([[1,0,-q[0]/q[2],0],
                          [0,1,-q[1]/q[2],0],
                          [0,0,0,0],
                          [0,0,-q[3]/q[2],1]]).astype('float64')
    return dpdq

def landmark_init(z,pose,imu_T_cam,fsu,b,K):
    z0=fsu*b/(z[0]-z[2])
    x0,y0=(z0*np.linalg.inv(K)@np.array([z[0],z[1],1.]))[0:2]
    landmark_s=(pose@imu_T_cam@np.array([x0,y0,z0,1.]))[0:3]
    #print(landmark_s)
    return landmark_s
def landmark_init_test(z,pose,imu_T_cam,fsu,b,K):
    print(K)
    z0=fsu*b/(z[0]-z[2])
    print(np.linalg.inv(K).shape)
    print(np.array([z[0],z[1],1.]).shape)
    x0,y0=(z0*np.linalg.inv(K)@np.array([z[0],z[1],1.]))[0:2]
    landmark_s=(pose@imu_T_cam@np.array([x0,y0,z0,1.]))[0:3]
    #print(landmark_s)
    return landmark_s

def landmark_update(z,miu,miu_sigma,pose,oTi,Ks,P_trans,V):
    '''
    H_landmark is a 4*3 matrix
    '''
    q=oTi@np.linalg.inv(pose)@(np.hstack((miu,[1])))
    z_predict=Ks@pi_norm(q)
    H_landmark=Ks@pi_Jacobian(q)@oTi@np.linalg.inv(pose)@P_trans
    Kal=miu_sigma@(H_landmark.T)@np.linalg.inv((H_landmark@miu_sigma@(H_landmark.T)+V*np.eye(4)))
    miu=miu+Kal@(z-z_predict)
    miu_sigma=(np.eye(3)-Kal@H_landmark)@miu_sigma

    
    return miu,miu_sigma    
def pose_sigma_predict_EKF(tao,u,sigma,W_pose):
    '''
    tao is a scalar
    u is a 6 matrix
    sigma is a 6*6 matrix
    W is a 6*6 matrix
    '''
    miu_perturb=scipy.linalg.expm(-tao* util.axangle2adtwist(u))
    #print(miu_perturb.shape)
    exp_u=scipy.linalg.expm(-tao* util.axangle2adtwist(u))
    sigma_predict=exp_u@sigma@(exp_u.T)+W_pose
    #print(sigma_predict)


    return sigma_predict

def circle_hat(s):
    '''
    s is a 4 matrix
    '''
    A=np.array([[1,0,0,0,s[2],-s[1]],
                [0,1,0,-s[2],0,s[0]],
                [0,0,1,s[1],-s[0],0],
                [0,0,0,0,0,0]]).astype('float64')
    return A
    


'''
def get_H_pose(Ks,oTi,pose,m):
    q=np.linalg.inv(pose)@np.hstack((m,[1]))
    #mat_A=circle_hat(q)
    mat_B=-Ks@pi_Jacobian(oTi@q)@oTi@circle_hat(q)
    #print(mat_A)
    #print(mat_B)
    return mat_B
'''
'''
def get_Kal_pose(sigma,H,V):
    Kal_pose=sigma@(H.T)@np.linalg.inv(H@sigma@(H.T)+V*np.eye(4))
    return Kal_pose
'''
def get_H_z_pose(Ks,oTi,pose,m):
    q=np.linalg.inv(pose)@np.hstack((m,[1]))
    z_predict=Ks@pi_norm(oTi@q)
    H_pose=-Ks@pi_Jacobian(oTi@q)@oTi@circle_hat(q)
    return H_pose,z_predict

    
def get_error(Ks,oTi,pose,m,sigma,V,z):
    q=np.linalg.inv(pose)@np.hstack((m,[1]))
    z_predict=Ks@pi_norm(oTi@q)
    H_pose=-Ks@pi_Jacobian(oTi@q)@oTi@circle_hat(q)
    Kal_pose=sigma@(H_pose.T)@np.linalg.inv(H_pose@sigma@(H_pose.T)+V*np.eye(4))
    delta_twist=Kal_pose@(z-z_predict)
    delta_sigma=Kal_pose@H_pose
    #print(delta_sigma.shape)
    return delta_twist,delta_sigma
def get_error_test(Ks,oTi,pose,m,sigma,V,z):
    q=np.linalg.inv(pose)@np.hstack((m,[1]))
    z_predict=Ks@pi_norm(oTi@q)
    H_pose=-Ks@pi_Jacobian(oTi@q)@oTi@circle_hat(q)
    print(sigma)
    #print(H_pose@sigma@(H_pose.T)+V*np.eye(4))
    Kal_pose=sigma@(H_pose.T)@np.linalg.inv(H_pose@sigma@(H_pose.T)+V*np.eye(4))
    delta_twist=Kal_pose@(z-z_predict)
    delta_sigma=Kal_pose@H_pose
    #print(delta_sigma.shape)
    return delta_twist,delta_sigma

if __name__ == '__main__':
    print('Function library:')
    MAP=define_map(-40,40,-40,40, 1)
    v=np.array([[1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12]]).T

    A=np.arange(4)
    q=np.array([1,2,3,1])
    print(circle_hat(q))
