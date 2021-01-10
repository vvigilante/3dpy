import numpy as np
import cv2
EPS = 10e-4

class Camera:
    def __init__(self, pos_x, pos_y, pos_z, rot_r, rot_p, rot_y, f):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.rot_r = rot_r
        self.rot_p = rot_p
        self.rot_y = rot_y
        self.f = f
    def get_transformation(self):
        return np.array(
            [
                [1, 0, 0, -self.pos_x],
                [0, 1, 0, -self.pos_y],
                [0, 0, 1, -self.pos_z],
                [0, 0, 0, 1],
            ], dtype=float
        )


class Vertex:
    def __init__(self, x, y, z):
        self.pos = np.array([x,y,z,1])

    def transform(self, A):
        self.pos = np.matmul(A,self.pos)

def _compute_normal(vertices):
    A,B,C = [v.pos[0:3] for v in vertices]
    AB = B-A
    AC = C-A
    n = np.cross(AB,AC)
    n /= np.linalg.norm(n)
    print(n)
    return n

class Face():
    def __init__(self, vertices, color):
        self.update(vertices,color)

    def update(self, vertices, color):
        assert len(vertices)==3, 'only triangles are supported'
        self.vertices = vertices
        self.color = color
        self.normal = _compute_normal(self.vertices)
    
    def transform(self, A):
        for v in self.vertices:
            v.transform(A)
        self.normal = _compute_normal(self.vertices)

class UniformShader():
    def __init__(self, face:Face):
        self.face = face

    def __call__(self,x,y):
        return self.face.color


class Light():
    pass


class DirectionalLight(Light):
    def __init__(self, direction, intensity):
        self.direction = np.array(direction, dtype=float)
        assert self.direction.shape==(3,), 'Direction must be a vector (x,y,z)'
        self.direction /= np.linalg.norm(direction) # normalize
        self.intensity = intensity


class FlatShader():
    def __init__(self, face:Face, lights: list):
        self.face = face
        self.lights = lights
        self.color = np.zeros(3,)
        for light in self.lights:
            if DirectionalLight==type(light):
                dot=np.abs(np.dot(self.face.normal, light.direction))
                print(dot)
                self.color+= dot*light.intensity*np.array(self.face.color, dtype=float)
        self.color = tuple(self.color.astype(np.uint8))

    def __call__(self,x,y):
        return self.color


class World:
    def __init__(self, w, h):
        self.canvas_shape = (h,w,3)
        self.camera = Camera(0,0,-2.0,0,0,0,1)
        self.faces = []
        self.lights = []
        self.shader = FlatShader

    def load_object(self, faces:list):
        for face in faces:
            self.faces.append(face)

    def load_light(self, light:Light):
        self.lights.append(light)

    def render(self):
        canvas = np.zeros( self.canvas_shape, dtype=np.uint8 )
        for f in self.faces:
            self._render_face(canvas, f)
        return canvas

    def _project(self, vertex):
        pos = vertex.pos
        A = self.camera.get_transformation()
        pos = np.matmul(A, pos)
        x,y,z,_ = pos
        # TODO: transform according to camera
        u = self.camera.f*x/z
        v = self.camera.f*y/z
        # reference to the canvas size
        h,w,_ = self.canvas_shape
        u = u*w/2+ w/2
        v = v*h/2+ h/2
        return u,v


    def _render_face(self, canvas, face):
        points = [self._project(p) for p in face.vertices]
        points = np.array(points)
        idx_top = np.argmin(points[:,1]) # find top point
        idx_bottom = np.argmax(points[:,1]) # find bottom point
        (idx_other,) = {0,1,2} - {idx_top, idx_bottom} # remaining point
        T, B, O = points[idx_top], points[idx_bottom], points[idx_other]
        #print(T,B,O)
        line_tb = Line(T,B)
        line_to = Line(T,O)
        line_ob = Line(B,O)
        pixel_shader = self.shader(face, self.lights)
        if line_to.is_horizontal: # triangle flat on top
            _render_line_to_line(canvas, line_tb, line_ob, T[1], B[1], pixel_shader)
            print('flat top')
        elif line_ob.is_horizontal: # triangle flat on bottom
            _render_line_to_line(canvas, line_tb, line_to, T[1], B[1], pixel_shader)
            print('flat base')
        else:
            # T..                Generic triangle (T,O,B)
            #   \   .            
            #    \_____. O       Split TB at point M with y=Oy
            #   M'\    /         
            #      \  /          We'll draw it left to right, top to bottom
            #       \/B          TM to TO, MB to OB
            M = [line_tb.get_x(O[1]),O[1]]
            print('generic')
            _render_line_to_line(canvas, line_tb, line_to, T[1], M[1], pixel_shader)
            _render_line_to_line(canvas, line_tb, line_ob, M[1], B[1], pixel_shader)

def _render_line_to_line(canvas, line1, line2, y_start, y_end, shader):
    # enforce canvas boundaries
    y_start = round(max(y_start,0))
    y_end = round(min(y_end, canvas.shape[0]-1))
    # draw top to bottom
    for y in range(y_start, y_end):
        x1 = line1.get_x(y)
        x2 = line2.get_x(y)
        x_start,x_end= (x1,x2) if x1<x2 else (x2,x1)
        # enforce canvas boundaries
        x_start = round(max(x_start,0))
        x_end = round(min(x_end, canvas.shape[1]-1))
        #print(y, x_start, x_end)
        for x in range(x_start, x_end):
            canvas[int(y),int(x),:] = shader(x,y)

class Line():
    def __init__(self, p1, p2):
        # y1= mx1+ q
        # y2= mx2+ q
        
        # m = (y1-q)/x1
        # q = y2-mx2
        # m = (y1-y2+mx2)/x1
        # m (1- x2/x1) = (y1-y2)/x1
        # m (x1 - x2)/x1 = (y1-y2)/x1
        # m = dy/dx
        x1,y1 = p1
        x2,y2 = p2
        self.is_vertical = abs(x1-x2)<1
        self.is_horizontal = abs(y1-y2)<1
        print("y1-y2", abs(y1-y2), "   x1-x2", abs(x1-x2))
        dx = x1-x2
        dy = y1-y2
        if self.is_vertical:
            self.q = x2
            self.m = None # Infinite if vertical
            self.invm = 0
        else:
            self.m = dy / dx 
            self.invm = dx / dy
            self.q = y2-self.m*x2

    def get_y(self, x):
        return x*self.m + self.q
        
    def get_x(self, y):
        if self.m is None:
            return self.q
        else:
            return (y - self.q)*self.invm


def getrot(t): # rotate along y
    A = np.array([
        [ np.cos(t), 0, np.sin(t), 0],
        [         0, 1,          0, 0],
        [-np.sin(t), 0,  np.cos(t), 0],
        [         0, 0,          0, 1]
    ])
    return A

if __name__ == "__main__":
    w = World(640, 360)
    test_obj = [
        Face([Vertex(-0.2, 0.2, 0.0), Vertex(0.0, 0.2, 0.0), Vertex(0.0, -0.4, 0.0)], (0,0,200))
    ]
    w.load_object(test_obj)
    w.load_light(DirectionalLight((1,1,1),1))

    while True:
        canvas = w.render()
        cv2.imshow('3dpy', canvas)
        k = cv2.waitKey(50)
        if k == ord('q'):
            break
        elif k==ord('a'):
            A = getrot(-0.1)
            test_obj[0].transform(A)
        elif k==ord('d'):
            A = getrot(0.1)
            test_obj[0].transform(A)
