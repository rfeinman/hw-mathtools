"""
Homework #1, Math Tools
Reuben Feinman
reuben.feinman@nyu.edu
9/13/17

Since you cannot execute sections of code in Python like with MATLAB, we will
use a command line parameter 'p' to choose what problem to run. As an example,
to run the code for Problem 1, enter:

python hw1.py -p 1

Note: I use Python 2, and I'm not sure what parts are or are not compatible
with Python 3.
"""
import textwrap
import argparse
import numpy as np
import matplotlib.pylab as plt

np.random.seed(0)


def plotVec2(matrix):
    """
    Function for problem 1a
    """
    assert matrix.shape[0] == 2, "Matrix must have height 2."
    plt.figure()
    # First, plot the axes
    plt.plot([-1, 1], [0, 0], c='black')
    plt.plot([0, 0], [-1, 1], c='black')
    # Randomly generate colors for each line. Colors are 3D vectors of
    # RGB values between 0-1.
    colors = np.random.uniform(size=(matrix.shape[1], 3))
    # Now cycle through the column vectors and plot each
    for i in range(matrix.shape[1]):
        x, y = matrix[:, i]
        plt.plot([0, x], [0, y], c=colors[i])
        plt.scatter([x], [y], c=colors[i])
    plt.show()

def vecLenAngle(v1, v2):
    """
    Function for problem 1b. Angles are returned in radians. If one or more of
    the vectors has zero length, 'N/A' is returned for the angle.
    """
    # The magnitude is the sqrt of the sum of component squares
    mag1 = np.sqrt(np.sum(np.square(v1)))
    mag2 = np.sqrt(np.sum(np.square(v2)))
    # Dot product = mag1*mag2*cos(theta), where theta is the angle between v1
    # and v2. So theta = arccos(dot/(mag1*mag2))
    dot = np.sum(np.multiply(v1, v2))
    if mag1*mag2 == 0:
        theta = 'N/A'
    else:
        theta = np.arccos(dot/(mag1*mag2))

    return mag1, mag2, theta

def main(problem_num):
    if problem_num == 1:
        ### Problem 1

        ## System 1
        print("\nSYSTEM 1:")

        # Call the system S(). We have a vector 'a' and a vector 'b' defined as:
        a = np.array([2, 4])
        b = np.array([-1, -2])

        # Note that a = -2b
        assert np.array_equal(a, -2*b)
        print('a = -2b <-- verified.')

        # We also know the outputs of the system S(a) and S(b)
        S_a = np.array([-6, -6])
        S_b = np.array([3, 3])

        # Since a = -2b, then for the system to be linear, we must have that
        # S(a) = -2S(b) by the principle of superposition. Let's check:
        assert np.array_equal(S_a, -2*S_b)
        print('S(a) = -2S(b) <-- verified. \nSo S() is a linear system because '
              'it follows the principle of superposition.')
        print textwrap.dedent("""\
            Now let's find a matrix that could be associated with this system. 
            We know by dimension rules that this must be a 2x2 matrix. Let's 
            call it M:
             
            M =   [m_00, m_01]
                  [m_10, m_11]
                  
            We know that Ma = [-6, -6]. So we have 2m_00 + 4m_01 = -6. We also know
            that Mb = [3, 3], but this doesn't add any information because b is a
            linear combination of a. So one possible solution is m_00 = -1 and 
            m_01 = -1, but this is not unique because we have 2 variables and 1 
            equality. We also have 2m_10 + 4m_11 = -6, which leads us to possible
            values m_10 = -1, m_11 = -1. But these values are again not unique,
            for the same reasoning. So we have
            
            M = [-1, -1]
                [-1, -1]
                
            as a possible matrix for this system, but it is not unique (explained
            above).
            """)
        # Check that M works
        M = np.array([[-1, -1], [-1, -1]])
        assert np.array_equal(np.matmul(M, a), S_a)
        assert np.array_equal(np.matmul(M, b), S_b)

        ## System 2
        print('\nSYSTEM 2:')
        # Again we have 'a' and 'b'
        a = 1
        b = 2
        # Note that b = 2a
        assert b == 2*a
        print('b = 2a <-- verified.')
        # Check whether S(b) = 2S(a)
        S_a = np.array([1, 4])
        S_b = np.array([2, 6])
        try:
            assert np.array_equal(S_b, 2*S_a)
        except:
            print('S(a) != 2S(b). \nSo S() is not a linear system because it '
                  'does not follow the principle of superposition.')

        ## System 3
        print('\nSYSTEM 3:')
        # Now we have 3 vectors 'a,' 'b' and 'c'
        a = np.array([1, 1])
        b = np.array([1, -0.5])
        c = np.array([3, 0])
        # Note that c = a + 2b
        assert np.array_equal(c, a + 2*b)
        print('c = a + 2b <-- verified.')
        # Check whether S(c) = S(a) + 2S(b)
        S_a = np.array([4, -1])
        S_b = np.array([1, 3])
        S_c = np.array([6, 2])
        try:
            assert np.array_equal(S_c, S_a + 2*S_b)
        except:
            print('S(c) != S(a) + 2S(b). \nS() is not a linear system because '
                  'it does not follow the principle of superposition.')

        ## System 4
        print('\nSYSTEM 4:')
        # we have 2 vectors 'a' and 'b'
        a = np.asarray([2, 4])
        b = np.asarray([-2, 2])
        print textwrap.dedent("""\
            We can't use the superposition test here because we cannot 
            write one vector as a linear combination of the others. But we 
            know that the operator would have to be a 1x2 matrix M, so we can 
            do some basic algebra:
            2m_00 + 4m_01 = 0  -->  m_00 = -2m_01
            -2m_00 + 2m_01 = 3  -->  -2(-2m_01) + 2m_01 = 3
            -->  6m_01 = 3  -->  m_01 = 0.5
            m_00 = -2(0.5) = -1
            
            So the system is linear, because we have a 1x2 matrix M that satisfies
            Ma = 0 and Mb = 3. This matrix is M = [-1, 0.5] and I believe it
            is unique, because there was a unique solution when solving the
            system of equations to find it.
            """)
        M = np.asarray([[-1, 0.5]])
        # note: result of np.matmul(M, a) will be a 1-D vector, so we have to
        # index to get the scalar
        assert np.array_equal(np.matmul(M, a)[0], 0)
        assert np.array_equal(np.matmul(M, b)[0], 3)

        ## System 5
        print('\nSYSTEM 5:')
        print textwrap.dedent("""\
            This cannot be a linear system. In order for this to be a linear system,
            we would need to find a matrix M such that M0 = [1,2]. But any matrix
            multiplied by the scalar 0 will have a resulting matrix with entries
            that are all zeros. So this cannot be a linear system.
            """)

    elif problem_num == 2:
        ### Problem 2

        ## part (a)
        print('\nPART a')
        # Generate a random height-2 matrix and plot the columns using our
        # function (code can be found above)
        matrix = np.random.normal(size=(2, 4))
        print('Matrix:')
        print(matrix)
        #plotVec2(matrix)

        ## part (b)
        print('\nPART b')
        # Generate two random vectors and use our function to compute
        # the magnitudes & lengths
        v1 = np.random.normal(size=(2,))
        v2 = np.random.normal(size=(2,))
        print('Vector 1:')
        print(v1)
        print('Vector 2:')
        print(v2)
        mag1, mag2, theta = vecLenAngle(v1, v2)
        print('Magnitude 1: %0.2f' % mag1)
        print('Magnitude 2: %0.2f' % mag2)
        if type(theta) == str:
            print('Angle (radians): %s' % theta)
        else:
            print('Angle (radians): %0.2f' % theta)

        ## part (c)

    elif problem_num == 3:
        return
    elif problem_num == 4:
        return
    elif problem_num == 5:
        return
    else:
        raise Exception('problem_num command line parameter must be an integer '
                        'from 1 to 5, inclusive.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem_num',
                        help='The problem number to run.',
                        required=True, type=int)
    args = parser.parse_args()
    main(args.problem_num)