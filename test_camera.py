import sys,os
import numpy as np
import cv2

from py3d import Camera

AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2
def draw_camera(camera: Camera,  ax_horz:int, ax_vert:int):
    A = camera.get_transformation()
    s = 400
    scale = 10
    grid_step = 10*scale
    im = np.zeros((s,s,3), dtype=np.uint8)
    c_grid = [[50,50,50]]
    im[::grid_step, :, :] = c_grid
    im[:, ::grid_step, :] = c_grid
    axes_colors = [ (0,0,255), (0,255,0), (255,0,0) ]
    axes_letters = "xyz"
    im[s//2, :, :] = axes_colors[ax_vert]
    im[:, s//2, :] = axes_colors[ax_horz]
    cv2.putText(im, axes_letters[ax_horz], (s//2+10,s-20), cv2.FONT_HERSHEY_SIMPLEX, 1, axes_colors[ax_horz], 1)
    cv2.putText(im, axes_letters[ax_vert], (s-20,s//2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, axes_colors[ax_vert], 1)
    
    points = [
            [0,0,-1, 1],
            [0,0,1, 1],
            [1,0,0, 1],
            [-1,0,0, 1],
        ]
    def transf(p):
        p = np.array(p)
        p = np.matmul(A,p)
        p = p*scale + s/2
        return (round(p[ax_horz]), round(p[ax_vert]) ) # x,z plane
    points = [transf(p) for p in points]
    t = 2
    cv2.line(im, points[0], points[1], (255,255,255), t)
    cv2.line(im, points[0], points[2], (255,255,255), t)
    cv2.line(im, points[0], points[3], (100,255,255), t)
    return im


if __name__ == "__main__":
    k = 0
    rotax = 'x'
    while True:
        for i in np.linspace(0,2*np.pi)[:-1]:
            camera = Camera(0, 0, 5, i if rotax=='x' else 0, i if rotax=='y' else 0, i if rotax=='z' else 0, 10)
            cameraim_xz = draw_camera(camera, AXIS_X, AXIS_Z)
            cameraim_xy = draw_camera(camera, AXIS_X, AXIS_Y)
            cameraim_yz = draw_camera(camera, AXIS_Y, AXIS_Z)
            cameraim = np.vstack([cameraim_xz,cameraim_xy])
            tmp = np.zeros(cameraim_yz.shape, dtype=np.uint8)
            cv2.putText(tmp, 'Camera pos: %.2f %.2f %.2f'%(camera.pos_x, camera.pos_y, camera.pos_z), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
            cv2.putText(tmp, 'Camera rot: %.2f %.2f %.2f'%(camera.rot_x, camera.rot_y, camera.rot_z), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
            cv2.putText(tmp, 'Selected axis of rotation: %s'%rotax, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
            tmp = np.vstack([cameraim_yz,tmp])
            cameraim = np.hstack([cameraim,tmp])
            cv2.imshow('camera', cameraim)
            k=cv2.waitKey(50)
            if k==27 or k==ord('q'):
                sys.exit(0)
            elif k==ord('x'):
                rotax = 'x'
            elif k==ord('y'):
                rotax = 'y'
            elif k==ord('z'):
                rotax = 'z'