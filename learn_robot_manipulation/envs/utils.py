import numpy as np
import pybullet as p

class Utils:
    def __init__(self, pid, np_random):
        self.id = pid
        self.np_random = np_random

    def get_joint_positions(self, body_id, joint_ids):
        """
        Get the position of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.array[float[N]]: joint positions [rad]
        """
        if isinstance(joint_ids, int):
            return p.getJointState(body_id, joint_ids, physicsClientId=self.id)[0]
        return np.asarray([state[0] for state in p.getJointStates(body_id, joint_ids, physicsClientId=self.id)])

    def get_jacobian(self, robot_id, tool_id, q):
        r"""
        Return the full geometric Jacobian matrix :math:`J(q) = [J_{lin}(q)^T, J_{ang}(q)^T]^T`, such that:

        .. math:: v = [\dot{p}^T, \omega^T]^T = J(q) \dot{q}

        where :math:`\dot{p}` is the Cartesian linear velocity of the link, and :math:`\omega` is its angular velocity.

        Args:
            body_id (int): unique body id.
            link_id (int): link id.
            local_position (np.array[float[3]]): the point on the specified link to compute the Jacobian (in link
              local coordinates around its center of mass). If None, it will use the CoM position (in the link frame).
            q (np.array[float[N]]): joint positions of size N, where N is the number of DoFs.
            dq (np.array[float[N]]): joint velocities of size N, where N is the number of DoFs.
            des_ddq (np.array[float[N]]): desired joint accelerations of size N.

        Returns:
            np.array[float[6,N]], np.array[float[6,6+N]]: full geometric (linear and angular) Jacobian matrix. The
                number of columns depends if the base is fixed or floating.
        """
        # Note that q, dq, ddq have to be lists in PyBullet (it doesn't work with numpy arrays)
        if isinstance(q, np.ndarray):
            q = q.ravel().tolist()

        local_position = p.getLinkState(robot_id, tool_id, physicsClientId=self.id)[2]
        dq = [0]*len(q)
        # calculate full jacobian
        lin_jac, ang_jac = p.calculateJacobian(robot_id, tool_id, localPosition=local_position,
                                                      objPositions=list(q), objVelocities=dq, objAccelerations=dq, physicsClientId=self.id)
        return np.vstack((lin_jac, ang_jac))

    # def get_my_analytical_jacobian(self, robot_id, tool_id, q):
        
    #     if isinstance(q, np.ndarray):
    #         q = q.ravel().tolist()

    #     tool_orientation = np.asarray(p.getLinkState(robot_id, tool_id)[1])
    #     o = np.asarray(p.getEulerFromQuaternion(tool_orientation))
    #     (r,pi,y) = o
    #     B = np.array([[1, 0, np.sin(pi)],
    #                   [0, np.cos(r), -np.cos(pi)*np.sin(r)],
    #                   [0, np.sin(r), np.cos(pi)*np.cos(r)]])
    #     Bi = np.linalg.inv(B)
    #     # analytical jacobian
    #     temp = np.eye(6)
    #     temp[3:6,3:6] = Bi
    #     J = self.get_jacobian(robot_id, tool_id, q)
    #     Ja = np.matmul(temp, J)
    #     return Ja

    def get_jacobian_derivative_rpy_to_angular_velocity(self, rpy_angle):
        r"""
        Return the Jacobian that maps RPY angle rates to angular velocities, i.e. :math:`\omega = T(\phi) \dot{\phi}`.
        Warnings: :math:`T` is singular when the pitch angle :math:`\theta_p = \pm \frac{\pi}{2}`
        Args:
            rpy_angle (np.array[float[3]]): RPY Euler angles [rad]
        Returns:
            np.array[float[3,3]]: Jacobian matrix that maps RPY angle rates to angular velocities.
        """
        roll, pitch, yaw = rpy_angle
        T = np.array([[1., 0., np.sin(pitch)],
                      [0., np.cos(roll), -np.cos(pitch) * np.sin(roll)],
                      [0., np.sin(roll), np.cos(pitch) * np.cos(roll)]])
        return T

    def get_analytical_jacobian(self, jacobian, rpy_angle):
        r"""
        Return the analytical Jacobian :math:`J_{a}(q) = [J_{lin}(q), J_{\phi}(q)]^T`, which respects:
        .. math:: \dot{x} = [\dot{p}, \dot{\phi}]^T = J_{a}(q) \dot{q}
        where :math:`\dot{p}` is the Cartesian linear velocity of the link, and :math:`\phi` are the Euler angles
        representing the orientation of the link. In general, the derivative of the Euler angles is not equal to
        the angular velocity, i.e. :math:`\dot{\phi} \neq \omega`.
        The analytical and geometric Jacobian are related by the following expression:
        .. math::
            J_{a}(q) = \left[\begin{array}{cc}
                I_{3 \times 3} & 0_{3 \times 3} \\
                0_{3 \times 3} & T^{-1}(\phi)
                \end{array} \right] J(q)
        where :math:`T` is the matrix that respects: :math:`\omega = T(\phi) \dot{\phi}`.
        Warnings:
            - We assume that the Euler angles used are roll, pitch, yaw (RPY)
            - We currently compute the analytical Jacobian from the geometric Jacobian. If we assume that we use RPY
                Euler angles then T is singular when the pitch angle :math:`\theta_p = \pm \frac{\pi}{2}.
        Args:
            jacobian (np.array[float[6,N]], np.array[float[6,6+N]]): full geometric Jacobian.
            rpy_angle (np.array[float[3]]): RPY Euler angles
        Returns:
            np.array[float[6,N]], np.array[float[6,6+N]]: the full analytical Jacobian. The number of columns
                depends if the base is fixed or floating.
        """
        T = self.get_jacobian_derivative_rpy_to_angular_velocity(rpy_angle)
        Tinv = np.linalg.inv(T)
        Ja = np.vstack((np.hstack((np.identity(3), np.zeros((3, 3)))),
                        np.hstack((np.zeros((3, 3)), Tinv)))).dot(jacobian)
        return Ja

    def enable_force_torque_sensor(self, body_id, joint_ids):
        
        if isinstance(joint_ids, int):
            p.enableJointForceTorqueSensor(body_id, joint_ids, 1)
        else:
            for joint_id in joint_ids:
                p.enableJointForceTorqueSensor(body_id, joint_id, 1)

    def force_torque_sensing(self, body_id, joint_ids):
        """
        Return the joint reaction forces at the given joint. Note that the torque sensor must be enabled, otherwise
        it will always return [0,0,0,0,0,0].

        Args:
            body_id (int): unique body id.
            joint_ids (int, int[N]): joint id, or list of joint ids

        Returns:
            if 1 joint:
                np.array[float[6]]: joint reaction force (fx,fy,fz,mx,my,mz) [N,Nm]
            if multiple joints:
                np.array[float[N,6]]: joint reaction forces [N, Nm]
        """
        if isinstance(joint_ids, int):
            return np.asarray(p.getJointState(body_id, joint_ids, physicsClientId=self.id)[2])
        return np.asarray([state[2] for state in p.getJointStates(body_id, joint_ids, physicsClientId=self.id)])

    def quaternion_error(self, quat_des, quat_cur):
        r"""
        Compute the orientation (vector) error between the current and desired quaternion; that is, it is the difference
        between :math:`q_curr` and :math:`q_des`, which is given by: :math:`\Delta q = q_{curr}^{-1} q_{des}`.
        Only the vector part is returned which can be used in PD control.

        Args:
            quat_des (np.array[float[4]]): desired quaternion [x,y,z,w]
            quat_cur (np.array[float[4]]): current quaternion [x,y,z,w]

        Returns:
            np.array[float[3]]: vector error between the current and desired quaternion
        """
        diff = quat_cur[-1] * quat_des[:3] - quat_des[-1] * quat_cur[:3] - self.skew_matrix(quat_des[:3]).dot(quat_cur[:3])
        return diff

    def skew_matrix(self, vector):
        r"""
        Return the skew-symmetric matrix of the given vector, which allows to represents the cross product between the
        given vector and another vector, as the multiplication of the returned skew-symmetric matrix with the other
        vector.

        The skew-symmetric matrix from a 3D vector :math:`v=[x,y,z]` is given by:

        .. math::

            S(v) = \left[ \begin{array}{ccc} 0 & -z & y \\ z & 0 & -x \\ -y & x & 0 \\ \end{array} \right]

        It can be shown [2] that: :math:`\dot{R}(t) = \omega(t) \times R(t) = S(\omega(t)) R(t)`, where :math:`R(t)` is
        a rotation matrix that varies as time :math:`t` goes, :math:`\omega(t)` is the angular velocity vector of frame
        :math:`R(t) with respect to the reference frame at time :math:`t`, and :math:`S(.)` is the skew operation that
        returns the skew-symmetric matrix from the given vector.

        Args:
            vector (np.array[float[3]]): 3D vector

        Returns:
            np.array[float[3,3]]: skew-symmetric matrix

        References:
            - [1] Wikipedia: https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product
            - [2] "Robotics: Modelling, Planning and Control" (sec 3.1.1), by Siciliano et al., 2010
        """
        x, y, z = np.array(vector).flatten()
        return np.array([[0., -z, y],
                        [z, 0., -x],
                        [-y, x, 0.]])

    def get_damped_least_squares_inverse(self, jacobian, damping_factor=0.01):
        r"""
        Return the damped least-squares (DLS) inverse, given by:

        .. math:: \hat{J} = J^T (JJ^T + k^2 I)^{-1}

        which can then be used to get joint velocities :math:`\dot{q}` from the cartesian velocities :math:`v`, using
        :math:`\dot{q} = \hat{J} v`.

        Args:
            jacobian (np.array[float[D,N]]): Jacobian matrix
            damping_factor (float): damping factor

        Returns:
            np.array[float[N,D]]: DLS inverse matrix
        """
        J, k = jacobian, damping_factor
        return J.T.dot(np.linalg.inv(J.dot(J.T) + k**2 * np.identity(J.shape[0])))

    def map_action_values(self, output_lower_limit, output_upper_limit, action_value):
        input_upper_limit, input_lower_limit = self.action_space.high[0], self.action_space.low[0]
        output = output_lower_limit + ((output_upper_limit - output_lower_limit) / (input_upper_limit - input_lower_limit)) * (action_value - input_lower_limit)
        return output

    def euclidian_distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)    

    def tool_position(self, body_id, tool_id):
        return np.asarray(p.getLinkState(body_id, tool_id, physicsClientId=self.id)[0])        

    def tool_orientation(self, body_id, tool_id):
        return np.asarray(p.getLinkState(body_id, tool_id, physicsClientId=self.id)[1])      