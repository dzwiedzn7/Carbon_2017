"""
This should be packege for backfiltered reconstruction
the penetration of the primary beam, ready to use for non-python programmers
I hope it will work

todo - rhodis
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import itertools
from scipy.stats import norm
from matplotlib.gridspec import GridSpec
from numpy import linalg as LA

import fire #package required to use below function from terminal level

def get_data(name):
    """
    basic function for loading data as an array
    required arguments:
    -name of file with your data [must containt 32 elemtents per each row]
    """
    data = np.loadtxt(name)
    return data

def choose_origin_mass(name,A):
    """
    this function load selected data an array based on atomic mass A
    required arguments:
    -name of file with your data [must containt 32 elemtents per each row]
    -atomic mass
    """
    data = np.loadtxt(name)
    origin = np.array([i for i in data if i[18] == A])
    return origin

def choose_origin_number(name,A):
    """
    this function load selected data an array based on atomic number Z
    required arguments:
    -name of file with your data [must containt 32 elemtents per each row]
    -atomic number
    """
    data = np.loadtxt(name)
    origin = np.array([i for i in data if i[17] == A])
    return origin

def angle(test1):
    """
    Calculate angele between the ray ending at the origin and passing through
    the point (1,0), and the ray ending at the origin and passing through
    the point of proton track at the exit point of the volume containing the dicom
    required arguments:
    -single event (row) from data array
    """
    vec1 = np.array([test1[9],test1[10]])
    return np.arctan2(vec1[1], vec1[0]) * 180 / np.pi


def data_proton(name,A = None):
    """
    loading data as an array w/o given additional parameter,from terminal level
    this function after call with prompt question  about data selection
    -ALL means that you are loading all events with no selection
    -A means that you select only events with given atomic mass
    -Z means that you select only events with given atomic number

    required arguments:
    -name of file with your data [must containt 32 elemtents per each row]
    -(optional)integeer that stands for atomic mass OR number [dependend from your previous choice (A/Z)]
    """
    choice = raw_input('Choose your data type[ALL/A/Z]: ')
    if choice == 'ALL':
        data = get_data(name)
    elif choice == 'A':
        data = choose_origin_mass(name,A)
    elif choice == 'Z':
        data = choose_origin_number(name,A)
    return data

def detectors(name,how_many,A=None):
    """
    Setting detectors
    this function will filter proton beam with dependence of their angle
    for example
    if you will chose to have 1 detector in Position given by angles 60 alpha
    and 30 beta its mean that as a result you will get an array of events that contain
    only those protons incidence between 60 and 30 degrees.
    You can choose how many detectors you want,each with any angle you want.

    required arguments:
    -name of file with your data [must containt 32 elemtents per each row]
    -number of detectors you want
    -(optional)integeer that stands for atomic mass OR number [dependend from your previous choice (A/Z)]

    remeber to pass arguments in this order!!!
    """
    data = data_proton(name,A)
    det =[]
    if how_many > 0:
        for i in range(1,how_many+1):
            alpha = int(raw_input('give alpha angle: '))
            beta = int(raw_input('give beta angle: '))
            detector = np.array(filter(lambda x:angle(x) < alpha and angle(x) > beta,data))
            det.append(detector)
        detector_full = np.concatenate(det,axis = 0)
    else:
        detector_full = data
    return detector_full

def dot_prod(name):
    """
    this function will give you an array of dot products of cx,cy,cz,cxprod,cyprod,czprod
    where cx,cy,cz are direction cosines of track at exit point and
    cxprod,cyprod,czprod are direction cosines at Production of current particle

    required arguments:
    -name of file with your data [must containt 32 elemtents per each row]
    """
    datax = np.loadtxt(name)
    dotss=[]
    for data in datax:
        dots = np.dot(data[12],data[22]) + np.dot(data[13],data[23]) + np.dot(data[14],data[24])
        dotss.append(dots)
    plt.hist(dotss,bins=100)
    plt.xlabel('Dot product of Ci ciprod')
    plt.ylabel('Frequency')
    plt.show()
    return dotss


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def line_intersect(p0,p1,p2,p3):
    """
    intersection algorith between two lines
    output:array containing  x,y intersection Position

    required arguments:
    -p0 is array of two values (x,y) cordinates of proton track at the exit point for first line
    -p1 is  array of two values (cx,cy) for first line
    -p0 is array of two values (x,y) cordinates of proton track at the exit point for second line
    -p1 is  array of two values (cx,cy) for second line
    """
    A1 = p1[1] - p0[1]
    B1 = p0[0] - p1[0]
    C1 = A1*p0[0] + B1 *p0[1]
    A2 = p3[1] - p2[1]
    B2 = p2[0] - p3[0]
    C2 = A2 * p2[0] + B2 *p2[1]
    den = A1*B2 - A2*B1
    if den == 0:
        print "Lines are paralel"
        x = None
        y = None
    else:
        x = (B2 * C1 - B1 * C2)/den
        y = (A1 * C2 - A2 * C1)/den
    return np.array([x,y])



def inter(name,how_many,div,A=None):
    """
    function to reconstruct primaty beam penetration,its Calculate every possible intersections between any two lines
    from data(no repeats) and then it Calculate Position of centroid,statistical error and standard deviation
    also is creating  scatter plots of intersection with additional plots of distibution by given axis
    required arguments:
    required arguments:
    -name of file with your data [must containt 32 elemtents per each row]
    -number of detectors you want
    -div which is integeer that you use to relieve RAM memory,by taking every div intersection.If you have a lot of RAM you can give 1
    -(optional)integeer that stands for atomic mass OR number [dependend from your previous choice (A/Z)]

    """
    detector = detectors(name,how_many,A)
    intersections = [line_intersect([test1[9],test1[10]],[test1[9]+test1[12],test1[10]+test1[13]],[test2[9],test2[10]],[test2[9]+test2[12],test2[10]+test2[13]]) for test1,test2 in itertools.combinations(detector[::div], 2)]
    inter = filter(lambda x:x[0] > -20. and x[0] <20. and x[1] < 20. and x[1] > -20. ,intersections)
    intersections = np.array(inter).T
    centroid_x = sum(intersections[0])/len(intersections[0])
    centroid_y = sum(intersections[1])/len(intersections[1])
    sem_x,sem_y = stats.sem(intersections[0], axis=None, ddof=0),stats.sem(intersections[1], axis=None, ddof=0)
    stdx,stdy = np.std(intersections[0]),np.std(intersections[1])
    print 'Pos X ' +'Pox Y ' +'sem X '+'sem Y '+'std X '+'std Y '
    print centroid_x,centroid_y,round(sem_x,3),round(sem_y,3),round(stdx,3),round(stdy,3)
    gs = GridSpec(4,4)
    fig = plt.figure()
    ax_joint = fig.add_subplot(gs[1:4,0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[1:4,3])
    ax_marg_x.hist(intersections[0],bins=50)
    ax_marg_y.hist(intersections[1],bins=50,orientation='horizontal')
    ax_joint.scatter(intersections[0],intersections[1])
    ax_joint.scatter(centroid_x,centroid_y)
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    ax_joint.set_xlabel('X Position')
    ax_joint.set_ylabel('Y Position')
    ax_marg_y.set_xlabel('X Freaquency')
    ax_marg_x.set_ylabel('Y Freaquency')
    plt.plot()
    plt.show()

def inter_chunks(name,how_many,div,chunk,A=None):
    """
    Similar function to inter but its divide all Calculated intersections into chunks,and every another calculation
    and plot is made for this particular chunk
    required arguments:
    -name of file with your data [must containt 32 elemtents per each row]
    -number of detectors you want
    -div which is integeer that you use to relieve RAM memory,by taking every div intersection.If you have a lot of RAM you can give 1
    -size of chunk
    -(optional)integeer that stands for atomic mass OR number [dependend from your previous choice (A/Z)]

    """
    detector = detectors(name,how_many,A)
    intersections = [line_intersect([test1[9],test1[10]],[test1[9]+test1[12],test1[10]+test1[13]],[test2[9],test2[10]],[test2[9]+test2[12],test2[10]+test2[13]]) for test1,test2 in itertools.combinations(detector[::div], 2)]
    inter = filter(lambda x:x[0] > -20. and x[0] <20. and x[1] < 20. and x[1] > -20. ,intersections)
    data = [i for i in chunks(inter,chunk)]
    print 'x','y','delta x','delta y'
    for intersection in data:
        intersection = np.array(intersection).T
        centroid_x = sum(intersection[0])/len(intersection[0])
        centroid_y = sum(intersection[1])/len(intersection[1])
        sem_x,sem_y = stats.sem(intersection[0], axis=None, ddof=0),stats.sem(intersection[1], axis=None, ddof=0)
        stdx,stdy = np.std(intersection[0]),np.std(intersection[1])
        print 'Pos X ' +'Pox Y ' +'sem X '+'sem Y '+'std X '+'std Y '
        print centroid_x,centroid_y,round(sem_x,3),round(sem_y,3),round(stdx,3),round(stdy,3)
        gs = GridSpec(4,4)
        fig = plt.figure()
        ax_joint = fig.add_subplot(gs[1:4,0:3])
        ax_marg_x = fig.add_subplot(gs[0,0:3])
        ax_marg_y = fig.add_subplot(gs[1:4,3])
        ax_marg_x.hist(intersection[0],bins=50)
        ax_marg_y.hist(intersection[1],bins=50,orientation='horizontal')
        ax_joint.scatter(intersection[0],intersection[1])
        ax_joint.scatter(centroid_x,centroid_y)
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)
        ax_joint.set_xlabel('X Position')
        ax_joint.set_ylabel('Y Position')
        ax_marg_y.set_xlabel('X Freaquency')
        ax_marg_x.set_ylabel('Y Freaquency')
        plt.plot()
        plt.show()

def energy(name):
    """
    function for ploting diffrance of kinetic energy of particle and kinetic energy at production
    in dependece of crossed material,with density distribution
    required arguments
    -name of file with your data [must containt 32 elemtents per each row]
    """
    data = np.loadtxt(name)
    ekin = []
    ekin_prod = []
    rhodis = []
    for i in data:
        ekin.append(i[15])
        ekin_prod.append(i[25])
        rhodis.append(i[29])
    energy_diff = np.array(ekin_prod) - np.array(ekin)
    xy = np.vstack([rhodis,energy_diff])
    z = stats.gaussian_kde(xy)(xy)
    plt.scatter(rhodis,energy_diff,s=10,c=z,edgecolor='')
    plt.xlabel('Distance of crossed material')
    plt.ylabel('diffrance of energy ekin_prod-ekin')
    plt.show()



if __name__ == '__main__':
    """
    To run any function
    -run terminal in directory that contain this script and data
    -type command:
    python carbon_2017.py [function name] [arguments required by function]
    ORDER OF GIVEN ARGUMENTS IS IMPORTANT
    """
    fire.Fire()
