import numpy as np
from PIL import Image

class Camera:
    def __init__(self, pos_x, pos_y, pos_z, rot_r, rot_p, rot_y, f):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.rot_r = rot_r
        self.rot_p = rot_p
        self.rot_y = rot_y
        self.f = f


class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def pos(self):
        return (self.x, self.y, self.z)

class Face():
    def __init__(self, vertices, color):
        self.vertices = vertices
        self.color = color


class UniformShader():
    def __init__(self, face:Face):
        self.face = face

    def __call__(self,x,y):
        return self.face.color

class World:
    def __init__(self, w, h):
        self.canvas_shape = (h,w,3)
        self.camera = Camera(0,0,0,0,0,0,1)
        self.faces = []
        self.lights = []
        self.shader = UniformShader

    def load_object(self, faces):
        for face in faces:
            self.faces.append(face)

    def render(self):
        canvas = np.zeros( self.canvas_shape, dtype=np.uint8 )
        for f in self.faces:
            self._render_face(canvas, f)
        return canvas

    def show(self):
        canvas = self.render()
        Image.fromarray(canvas).show()

    def _project(self, vertex):
        x,y,z = vertex.pos()
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
        pixel_shader = self.shader(face)
        if line_to.is_horizontal: # triangle flat on top
            _render_line_to_line(canvas, line_tb, line_ob, T[1], B[1], pixel_shader)
        elif line_ob.is_horizontal: # triangle flat on bottom
            _render_line_to_line(canvas, line_tb, line_to, T[1], B[1], pixel_shader)
        else:
            # T..                Generic triangle (T,O,B)
            #   \   .            
            #    \_____. O       Split TB at point M with y=Oy
            #   M'\    /         
            #      \  /          We'll draw it left to right, top to bottom
            #       \/B          TM to TO, MB to OB
            M = [line_tb.get_x(O[1]),O[1]]
            #print('generic')
            _render_line_to_line(canvas, line_tb, line_to, T[1], M[1], pixel_shader)
            _render_line_to_line(canvas, line_tb, line_ob, M[1], B[1], pixel_shader)

def _render_line_to_line(canvas, line1, line2, y_start, y_end, shader):
    # enforce canvas boundaries
    y_start = int(max(y_start,0))
    y_end = int(min(y_end, canvas.shape[0]-1))
    # draw top to bottom
    for y in range(y_start, y_end):
        x1 = line1.get_x(y)
        x2 = line2.get_x(y)
        x_start,x_end= (x1,x2) if x1<x2 else (x2,x1)
        #print(y, x_start, x_end)
        for x in range(int(x_start), int(x_end)):
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
        self.is_vertical = x1==x2
        self.is_horizontal = y1==y2
        dx = x1-x2
        dy = y1-y2
        if self.is_vertical:
            self.q = x2
            self.m = None
            self.invm = 0
        else:
            self.m = dy / dx # Infinite if vertical
            self.invm = dx / dy
            self.q = y2-self.m*x2

    def get_y(self, x):
        return x*self.m + self.q
        
    def get_x(self, y):
        return (y - self.q)*self.invm


if __name__ == "__main__":
    w = World(640, 360)
    test_obj = [
        Face([Vertex(-0.2, -0.2, 1.0), Vertex(0.4, 0.4, 1.0), Vertex(0.0, -0.4, 0.2)], (0,0,200))
    ]
    w.load_object(test_obj)
    w.show()