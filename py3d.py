import sys, os
import numpy as np
import cv2
import functools
from tqdm import tqdm
from time import time

def point_equal(A,B):
    return abs(A[0]-B[0])<1 and abs(A[1]-B[1])<1

def compute_transformation(rx:float,ry:float,rz:float,px:float,py:float,pz:float):
    cx = np.cos(rx)
    cy = np.cos(ry)
    cz = np.cos(rz)
    sx = np.sin(rx)
    sy = np.sin(ry)
    sz = np.sin(rz)
    return np.array(
        [
            [ cy*cz,  sx*sy*cz+cx*sz, -cx*sy*cz+sx*sz, px],
            [-cy*sz, -sx*sy*sz+cx*cz,  cx*sy*sz+sx*cz, py],
            [    sy,          -sx*cy,           cx*cy, pz],
            [     0,               0,               0,  1],
        ], dtype=float
    )

class Camera(object):
    def __init__(self, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, f):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z
        self.f = f

    def __setattr__(self, name, value):
        super(Camera, self).__setattr__(name, value)
        if 'rot_x' in self.__dict__ and 'rot_y' in self.__dict__ and 'rot_z' in self.__dict__ and 'pos_x' in self.__dict__ and 'pos_y' in self.__dict__ and 'pos_z' in self.__dict__:
            self.__dict__['_A'] = self._compute_transformation()

    def get_transformation(self):
        return self._A

    def _compute_transformation(self):
        A = compute_transformation(self.rot_x, self.rot_y, self.rot_z, self.pos_x, self.pos_y, self.pos_z)
        return A


class Vertex:
    def __init__(self, pos):
        if len(pos)==3:
            self.pos = np.array(list(pos)+[1])
        else:
            assert len(pos)==4
            self.pos = np.array(pos, dtype=float)
            self.pos[3] = 1
    
    def transform(self, A):
        return Vertex( np.matmul(A,self.pos) )

    def __getattr__(self, name):
        if name=='x':
            return self.pos[0]
        elif name=='y':
            return self.pos[1]
        elif name=='z':
            return self.pos[2]
        else:
            raise AttributeError

def _compute_normal(vertices):
    A,B,C = vertices
    AB = B-A
    AC = C-A
    n = np.cross(AB,AC)
    n /= np.linalg.norm(n)
    return n

class Face():
    def __init__(self, vertices, color):
        assert len(vertices)==3, 'only triangles are supported'
        self.vertices = vertices
        self.color = color
        self.normal = _compute_normal([v.pos[0:3] for v in self.vertices])
    
    def transform(self, A):
        transformed_vertices = []
        for v in self.vertices:
            transformed_vertices.append( v.transform(A) )
        return Face(transformed_vertices, self.color)

class UniformShader():
    def __init__(self, face:Face, lights:list, points: list):
        self.face = face
        self.normal = _compute_normal(points)
        self.wireframe=(255,255,255)
        self.color_back = tuple(np.array(self.face.color, dtype=int)/4)

    def __call__(self,x,y):
        if self.normal[2] > 0:
            return self.face.color
        else:
            return self.color_back


class Light():
    pass


class DirectionalLight(Light):
    def __init__(self, direction, intensity):
        self.set_direction(direction)
        self.intensity = intensity

    def set_direction(self, direction):
        self.direction = np.array(direction, dtype=float)
        assert self.direction.shape==(3,), 'Direction must be a vector (x,y,z)'
        self.direction /= np.linalg.norm(direction) # normalize

class WireframeShader():
    def __init__(self, face:Face, lights: list, points=None):
        self.wireframe=(face.color)
    def __call__(self,x,y):
        return None


class FlatShader():
    def __init__(self, face:Face, lights: list, points: list):
        self.face = face
        self.lights = lights
        self.wireframe=(255,255,255)
        self.ambient = 0.1 # TODO move
        self.color = np.zeros(3,)
        self.normal = self.face.normal #_compute_normal(points)
        for light in self.lights:
            if DirectionalLight==type(light):
                dot=np.max([0,np.dot(self.normal, light.direction)])
                #dot=np.abs(np.dot(self.normal, light.direction))
                #print(light.direction, self.normal, dot)
                self.color+= dot*light.intensity*np.array(self.face.color, dtype=float)
        self.color = self.color*(1-self.ambient) + self.ambient*np.array(self.face.color, dtype=float)
        self.color = tuple(self.color.astype(np.uint8))

    def __call__(self,x,y):
        return self.color

class FlatShaderNoWireframe(FlatShader):
    def __init__(self, *args, **kwargs):
        super(FlatShaderNoWireframe, self).__init__(*args, **kwargs)
        self.wireframe=None




class World:
    def __init__(self, w, h, shader):
        self.canvas_shape = [h,w]
        self.zbuf = None
        self.camera = Camera(0,-1.0,-2.0,0,0,0,1.0)
        self.faces = []
        self.lights = []
        self.shader = shader

    def load_object(self, faces:list):
        for face in faces:
            self.faces.append(face)

    def load_light(self, light:Light):
        self.lights.append(light)

    def render(self):
        zbuf = np.zeros(self.canvas_shape+[1], dtype=float)
        canvas = np.zeros( self.canvas_shape+[3], dtype=np.uint8 )
        for f in self.faces:
            self._render_face(canvas, zbuf, f)
        #for f in faces:
        #    for p in f.vertices:
        #        u,v,z = self._project(p)
        #        e = max(0, 255-min(255, (z-200.2)*300))
        #        cv2.putText(canvas, '%.2f'%z, (int(u),int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (e,e,e), 1)

        return canvas

    def _project(self, vertex):
        pos = vertex.pos
        A = self.camera.get_transformation()
        pos = np.matmul(A, pos)
        x,y,z,_ = pos
        x1 = self.camera.f*x/z
        y1 = self.camera.f*y/z
        # reference to the canvas size
        h,w = self.canvas_shape
        u = x1*w/2+ w/2
        v = y1*h/2+ h/2
        return u,v, z


    def _render_face(self, canvas, zbuf, face):
        points = [self._project(p) for p in face.vertices]
        points = np.array(points)
        idx_top = np.argmin(points[:,1]) # find top point
        idx_bottom = np.argmax(points[:,1]) # find bottom point
        if idx_top == idx_bottom: #, "triangle is a line %s"%(str(points))
            return

        (idx_other,) = {0,1,2} - {idx_top, idx_bottom} # remaining point
        T, B, O = points[idx_top], points[idx_bottom], points[idx_other]
        if point_equal(T,B) or point_equal(B,O) or point_equal(O,T): #triangle is a line, do not draw
            return 
        #print(T,B,O)
        line_tb = Line(T,B)
        line_to = Line(T,O)
        line_ob = Line(B,O)
        pixel_shader = self.shader(face, self.lights, points)
        if line_to.is_horizontal and line_ob.is_horizontal: # triangle is a line, do not draw
            pass
        elif line_to.is_horizontal: # triangle flat on top
            assert not line_ob.is_horizontal
            _render_line_to_line(canvas, zbuf, line_tb, line_ob, T[1], B[1], pixel_shader)
            #print('flat top')
        elif line_ob.is_horizontal: # triangle flat on bottom
            _render_line_to_line(canvas, zbuf, line_tb, line_to, T[1], B[1], pixel_shader)
            #print('flat base')
        else:
            # T..                Generic triangle (T,O,B)
            #   \   .            
            #    \_____. O       Split TB at point M with y=Oy
            #   M'\    /         
            #      \  /          We'll draw it left to right, top to bottom
            #       \/B          TM to TO, MB to OB
            M = [line_tb.get_x(O[1]),O[1]]
            #print('generic')
            _render_line_to_line(canvas, zbuf, line_tb, line_to, T[1], M[1], pixel_shader)
            _render_line_to_line(canvas, zbuf, line_tb, line_ob, M[1], B[1], pixel_shader)

def clip(num, min, max):
    if num<min: return min
    if num>max: return max
    return num

def _render_line_to_line(canvas, zbuf, line1, line2, y_start, y_end, shader):
    # enforce canvas boundaries
    y_start = round(clip(y_start,0, canvas.shape[0]-1))
    y_end = round(clip(y_end,0, canvas.shape[0]-1))
    # draw top to bottom
    for y in range(y_start, y_end):
        x1 = line1.get_x(y)
        x2 = line2.get_x(y)
            
        line_start,x_start,line_end,x_end= (line1,x1,line2,x2) if x1<x2 else (line2,x2,line1,x1)
        
        z_start = line_start.get_z(y)
        z_end = line_end.get_z(y)
        def get_z(x):
            t = (x - x_start) / (x_end-x_start)
            return (1-t)*z_start + t*z_end

        # enforce canvas boundaries
        x_start = round(clip(x_start, 0, canvas.shape[1]-1))
        x_end = round(clip(x_end, 0, canvas.shape[1]-1))
        
        for x in range(x_start, x_end):
            invz = 1.0/get_z(x)
            if invz > zbuf[y,x]:
                zbuf[y,x] = invz
                c = shader(x,y)
                if c is not None:
                    canvas[y,x,:] = c
                if shader.wireframe and (x==x_start or x==x_end):
                    canvas[y,x,:] = shader.wireframe

    
AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

def show_projection(faces, camera: Camera, ax_horz:int, ax_vert:int, scale=10):
    s = 400
    im = np.zeros((s,s,3), np.uint8)
    axes_colors = [ (0,0,255), (0,255,0), (255,0,0) ]
    axes_letters = "xyz"
    im[s//2, :, :] = axes_colors[ax_vert]
    im[:, s//2, :] = axes_colors[ax_horz]
    cv2.putText(im, axes_letters[ax_horz], (s//2+10,s-20), cv2.FONT_HERSHEY_SIMPLEX, 1, axes_colors[ax_horz], 1)
    cv2.putText(im, axes_letters[ax_vert], (s-20,s//2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, axes_colors[ax_vert], 1)
    
    A = camera.get_transformation()
    for face in faces:
        vertices = list(face.vertices)+[face.vertices[0]]
        c = (255,255,255)
        t = 1
        def transf(v):
            p = v.pos
            p = np.matmul(A,p)
            p = p*scale + im.shape[0]/2
            return (round(p[ax_horz]), round(p[ax_vert]) ) # x,z plane
        for v1,v2 in zip(vertices[1:], vertices[:-1]):
            cv2.line(im, transf(v1), transf(v2), c,t)
    return im


class Line():
    def __init__(self, p1, p2):
        assert not point_equal(p1,p2), "Line is a point!"
        # y1= mx1+ q
        # y2= mx2+ q
        
        # m = (y1-q)/x1
        # q = y2-mx2
        # m = (y1-y2+mx2)/x1
        # m (1- x2/x1) = (y1-y2)/x1
        # m (x1 - x2)/x1 = (y1-y2)/x1
        # m = dy/dx
        x1,y1 = p1[:2]
        x2,y2 = p2[:2]
        self.is_vertical = abs(x1-x2)<1
        self.is_horizontal = abs(y1-y2)<1
        dx = x1-x2
        dy = y1-y2
        if self.is_vertical:
            self.q = x2
            self.m = None # Infinite if vertical
            self.invm = 0
        elif self.is_horizontal:
            self.q = y2
            self.invm = None # Infinite if horizontal
            self.m = 0
        else:
            self.m = dy / dx 
            self.invm = dx / dy
            self.q = y2-self.m*x2
        assert not (self.is_horizontal and self.is_vertical)

        self.p1 = p1
        self.p2 = p2

    def get_y(self, x):
        return x*self.m + self.q
        
    def get_x(self, y):
        if self.m is None:
            assert self.is_vertical
            return self.q
        else:
            if self.invm is None:
                assert self.is_horizontal
            return (y - self.q)*self.invm

    def get_z(self, y):
        if self.p1[1] < self.p1[2]:
            ya,za = self.p1[1:3] # ya is the smallest
            yb,zb = self.p2[1:3]
        else:
            ya,za = self.p2[1:3]
            yb,zb = self.p1[1:3]
        t = (y - ya) / (yb-ya) # t = 0 if y=ya -> then we return z=za
        return za*(1-t) + zb*t



def getroty(t):
    A = np.array([
        [ np.cos(t), 0, -np.sin(t),  0],
        [         0, 1,          0, 0],
        [ np.sin(t), 0,  np.cos(t), 0],
        [         0, 0,          0, 1]
    ], dtype=float)
    return A

def getrotx(t):
    A = np.array([
        [         1,          0,          0, 0],
        [         0,  np.cos(t),  np.sin(t), 0],
        [         0, -np.sin(t),  np.cos(t), 0],
        [         0,          0,          0, 1]
    ], dtype=float)
    return A



def get_quad(v0,v1,v2,v3, **kwargs):
    return [
            Face([v0,v1,v2], **kwargs),
            Face([v2,v3,v0], **kwargs),
    ]

class Obj():
    def __init__(self):
        self.faces=[]
    def __getitem__(self, i, **kwargs):
        return self.faces.__getitem__(i, **kwargs)

    def __iter__(self, **kwargs):
        return self.faces.__iter__(**kwargs)
    
    def transform(self, A):
        for f in self.faces:
            f = f.transform(A) ##TODO < not working

class Cube(Obj):
    def __init__(self, d=0.5):
        super(Cube, self).__init__()
        vertices = [
            Vertex((-d, -d, -d)),
            Vertex((-d, -d,  d)),
            Vertex((-d,  d,  d)),
            Vertex((-d,  d, -d)),
            Vertex(( d, -d, -d)),
            Vertex(( d, -d,  d)),
            Vertex(( d,  d,  d)),
            Vertex(( d,  d, -d)),
        ]
        color = [200,200,200]
        self.faces = []
        self.faces += get_quad(vertices[0], vertices[1], vertices[2], vertices[3], color=color)
        self.faces += get_quad(vertices[4], vertices[5], vertices[6], vertices[7], color=color)
        self.faces += get_quad(vertices[0], vertices[4], vertices[5], vertices[1], color=color)
        self.faces += get_quad(vertices[2], vertices[6], vertices[7], vertices[3], color=color)
        self.faces += get_quad(vertices[1], vertices[5], vertices[6], vertices[2], color=color)
        self.faces += get_quad(vertices[0], vertices[3], vertices[7], vertices[4], color=color)
    


class STL(Obj):
    def __init__(self, path, color=[200,200,200]):
        super(STL, self).__init__()
        self.faces = []
        with open(path) as f:
            curface = None
            for line in f:
                line = line.split(' ')
                line[0]=line[0].strip()
                if line[0]=='facet':
                    curface = []
                elif line[0]=='endfacet':
                    assert len(curface)==3, "Only triangles are supported"
                    self.faces.append( Face(curface, color=color) )
                    curface = None
                elif line[0]=='vertex':
                    vertex = [float(v) for v in line[1:]]
                    curface.append(Vertex(vertex) )
        print("Loaded %d faces."%len(self.faces))
    




def main_interactive():
    w = World(360, 360, FlatShaderNoWireframe)
    #test_obj = Cube()
    test_obj = STL('Suzanne.stl')
    w.load_object(test_obj)
    w.load_light(DirectionalLight((1,-1,-1),1))
    w.camera.f = 5.0
    w.camera.pos_z = 7
    w.camera.pos_y = 0
    w.camera.rot_x = -1.5
    w.camera.rot_y = 0.0
    w.camera.rot_z = 0.0
    while True:
        '''
        scale=20
        xz = show_projection(test_obj, w.camera, AXIS_X,AXIS_Z, scale=scale)
        xy = show_projection(test_obj, w.camera, AXIS_X,AXIS_Y, scale=scale)
        yz = show_projection(test_obj, w.camera, AXIS_Y,AXIS_Z, scale=scale)
        proj = np.vstack([xz,xy])
        tmp = np.zeros(yz.shape, dtype=np.uint8)
        tmp = np.vstack([yz,tmp])
        proj = np.hstack([proj,tmp])
        cv2.imshow('projection', proj)
        '''
        t_start = time()
        canvas = w.render()
        t_elapsed = time()-t_start
        cv2.putText(canvas, 'Camera: %.2f %.2f %.2f %.2f %.2f %.2f'%(w.camera.pos_x, w.camera.pos_y, w.camera.pos_z, w.camera.rot_x, w.camera.rot_y, w.camera.rot_z), (14,14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(canvas, 'Time: %.2fs'%t_elapsed, (14,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow('3dpy', canvas)
        k = cv2.waitKey(5)
        rotq = np.pi/8
        if k == ord('q'):
            break
        elif k==ord('a'):
            A = getroty(-rotq)
            test_obj.transform(A)
        elif k==ord('d'):
            A = getroty(rotq)
            test_obj.transform(A)
        elif k==ord('w'):
            A = getrotx(-rotq)
            test_obj.transform(A)
        elif k==ord('s'):
            A = getrotx(rotq)
            test_obj.transform(A)
        elif k==ord('l'):
            w.camera.pos_x+=0.1
        elif k==ord('j'):
            w.camera.pos_x-=0.1
        elif k==ord('k'):
            w.camera.pos_y+=0.1
        elif k==ord('i'):
            w.camera.pos_y-=0.1
        elif k==ord('o'):
            w.camera.pos_z-=0.1
        elif k==ord('p'):
            w.camera.pos_z+=0.1
        elif k==ord('h'):
            w.camera.rot_x-=rotq
        elif k==ord('n'):
            w.camera.rot_x+=rotq
        elif k==ord('b'):
            w.camera.rot_y-=rotq
        elif k==ord('m'):
            w.camera.rot_y+=rotq

def main_batch():
    w = World(360, 360, FlatShaderNoWireframe)
    #test_obj = Cube()
    test_obj = STL('Suzanne.stl')
    test_light = DirectionalLight((1,-1,-1),1)
    w.load_object(test_obj)
    w.load_light(test_light)
    w.camera.f = 5.0
    w.camera.pos_z = 8
    w.camera.pos_y = 0
    w.camera.rot_x = -1.5
    w.camera.rot_y = 0.0
    w.camera.rot_z = 0.0
    for i,v in enumerate(tqdm(np.linspace(0,2*np.pi, num=96)[:-1])):
        w.camera.rot_y = v
        d = np.array([1,-1,-1, 1], dtype=float)
        lrx,lry,lrz = 0,0,v
        d = np.matmul( compute_transformation(lrx,lry,lrz, 0,0,0),  d)
        test_light.set_direction(d[:3])
        canvas = w.render()
        #cv2.putText(canvas, 'Cam: %.2f %.2f %.2f Light: %.2f %.2f %.2f F%02d'%(w.camera.rot_x, w.camera.rot_y, w.camera.rot_z, lrx,lry,lrz, i), (14,14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        canvas=canvas[60:300,:]
        cv2.imwrite('out/f%03d.png'%i, canvas)
    os.system('ffmpeg -v warning -i out/f%03d.png -vf "palettegen" -y out/palette.png')
    os.system('ffmpeg  -framerate 12 -v warning -i out/f%03d.png -i out/palette.png -lavfi "[0:v] paletteuse" -y "outtest.gif"')
    #os.system('rm -f out/f???.png')


if __name__ == "__main__":
    main_interactive()
    #main_batch()
