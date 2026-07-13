"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
def prod(shape):
    # TODO: Multiply together the elements of a shape tuple to get the total number of elements.
    result = 1
    for dim in shape: 
        result *= dim 
    return int(result)

# Step 2 - argsort
def argsort(values):
    # TODO: Return the indices that would sort values in ascending order.
    return sorted(range(len(values)), key = lambda i: values[i])

# Step 3 - make_op_enums
def make_op_enums():
    # TODO: create four enum classes naming every supported operation kind
    UnaryOps = enum.Enum('UnaryOps', ['NEG', 'RELU', 'LOG', 'EXP', 'SQRT', 'SIGMOID'])
    BinaryOps = enum.Enum('BinaryOps', ['ADD', 'SUB', 'MUL', 'DIV', 'CMPLT', 'MAX'])
    ReduceOps = enum.Enum('ReduceOps', ['SUM', 'MAX'])
    MovementOps = enum.Enum('MovementOps', ['RESHAPE', 'EXPAND', 'PERMUTE'])
    return (UnaryOps, BinaryOps, ReduceOps, MovementOps)

# Step 4 - LazyBuffer
class LazyBuffer:
    def __init__(self, np_array):
        self._np = np.asarray(np_array)
        self.shape = tuple(int(d) for d in self._np.shape)        
        self.dtype = self._np.dtype

    def __array__(self, dtype=None):
        return np.asarray(self._np, dtype=dtype)

    def __float__(self):
        return float(self._np)

    def __repr__(self):
        return repr(self._np)

    def __str__(self):
        return str(self._np)

# Step 5 - lazybuffer_const
def const(value, shape):
    # TODO: Create a new LazyBuffer of the given shape filled with a constant value.
    arr = np.full(shape, value, dtype = np.float32)
    return LazyBuffer(arr)
LazyBuffer.const = staticmethod(const)

# Step 6 - rand
def rand(shape, seed=None):
    # TODO: return a LazyBuffer of uniform random floats in [0, 1) with given shape
    """Theory: PRNG: Pseudo Random Number Generator"""
    rng = np.random.RandomState(seed)
    data = rng.random(shape).astype(np.float32)
    return LazyBuffer(data)

# Step 7 - lazybuffer_unary_e
import enum 
def e(self, op, *srcs):
    # TODO: apply a unary elementwise op (NEG, RELU, LOG, EXP, SQRT, SIGMOID)
    x = self._np
    if op.name == 'NEG': 
        result=  -x 
    elif op.name == 'RELU': 
        result= np.maximum(x, 0) 
    elif op.name == 'EXP': 
        result= np.exp(x)
    elif op.name == 'SQRT': 
        result= np.sqrt(x)
    elif op.name == 'SIGMOID': 
        result= 1.0 / (1.0+np.exp(-x))
    elif op.name == 'LOG': 
        result= np.log(x)
    else: 
        raise ValueError(f"Operation name: {op.name} Does not Exist")
    return LazyBuffer(result)

LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e
def lazybuffer_binary_e(self, op, other):
    # TODO: apply a binary elementwise op between two LazyBuffers, return a new LazyBuffer
    a = self._np 
    b = other._np   
    
    if op.name == "ADD":
        res = a + b 
    elif op.name == "SUB": 
        res = a - b 
    elif op.name == "MUL": 
        res = a * b 
    elif op.name == "DIV": 
        res = a / b 
    elif op.name == "CMPLT": 
        res = (a<b).astype(a.dtype)
    elif op.name == "MAX": 
        res = np.maximum(a, b)
    else: 
        raise ValueError(f"Operation name: {op.name} Does not Exist")
    return LazyBuffer(res)

# Step 9 - lazybuffer_r
def r(self, op, axis):
    # TODO: reduce the underlying array along axis (SUM or MAX), keeping reduced dims as size 1
    name = getattr(op, 'name', None)
    if op.name == "SUM": 
        res = self._np.sum(axis = axis, keepdims = True)
    elif op.name == "MAX": 
        res = self._np.max(axis = axis, keepdims = True)
    else: 
        raise ValueError(f"Operation name: {op.name} Does not Exist")
    return LazyBuffer(res)

# Step 10 - lazybuffer_reshape
def reshape(self, new_shape):
    # TODO: return a new LazyBuffer with the array reshaped to new_shape
    return LazyBuffer(self._np.reshape(new_shape))

# Step 11 - lazybuffer_expand
def expand(self, new_shape):
    # TODO: broadcast this buffer's size-1 dims out to new_shape
    new_shape = tuple(int(d) for d in new_shape)
    x = self._np
    out = np.broadcast_to(x, new_shape).copy()
    return LazyBuffer(np.array(out))

# Step 12 - lazybuffer_permute
def permute(self, order):
    # TODO: return a new LazyBuffer with axes reordered according to order
    x = self._np 
    x = x.transpose(order)
    return LazyBuffer(x)

# Step 13 - Function
class Function:
    def __init__(self, *tensors):
        # TODO: record needs_input_grad, requires_grad, and parents for backprop

        self.needs_input_grad = [t.requires_grad for t in tensors]

        if any(f is True for f in self.needs_input_grad): 
            self.requires_grad = True 
        elif None in self.needs_input_grad: 
            self.requires_grad = None 
        else: 
            self.requires_grad = False 
        
        if self.requires_grad: 
            self.parents = tensors

# Step 14 - function_forward_backward_stubs
def function_forward_backward_stubs():
    # TODO: attach forward and backward stubs to Function that raise NotImplementedError
    def forward(self, *args, **kwargs): 
        raise NotImplementedError(f"forward not implemented for {type(self).__name__}")
    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"backward not implemented for {type(self).__name__}") 
    Function.forward = forward 
    Function.backward = backward 
    return Function

# Step 15 - apply
@classmethod
def apply(cls, *tensors, **kwargs):
    # Build the Function context
    ctx = cls(*tensors)

    # Extract LazyBuffers
    bufs = [t.lazydata for t in tensors]
    # Run the forward pass
    out_buf = ctx.forward(*bufs, **kwargs)
    # Wrap the output buffer in a Tensor
    result = Tensor(out_buf, requires_grad=ctx.requires_grad)
    # Link the autograd context
    if ctx.requires_grad:
        result._ctx = ctx
    return result


# Provided: attaches apply onto the Function base class. Leave this as-is.
for _obj in list(globals().values()):
    if isinstance(_obj, type):
        for _k in _obj.__mro__:
            if _k.__name__ == 'Function':
                _k.apply = apply

# Step 16 - Neg
import numpy as np 
class Neg(Function):
    def forward(self, x):
        # TODO: return a LazyBuffer holding the elementwise negation of x
        return LazyBuffer(-x._np)

    def backward(self, grad_output):
        # TODO: return the negated incoming gradient
        return LazyBuffer(-grad_output._np)

# Step 17 - Relu
class Relu(Function):
    def forward(self, x):
        # TODO: apply the rectified linear unit to lazy buffer x and cache the result
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        self.rel = x.e(UnaryOps.RELU)
        return self.rel 

    def backward(self, grad_output):
        # TODO: route the upstream gradient only through positions that were positive
        # Building a Zero LazyBuffer with the same shape as self.rel 
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        zero = LazyBuffer.const(0, self.rel._np.shape)
        mask = lazybuffer_binary_e(zero, BinaryOps.CMPLT, self.rel)
        return lazybuffer_binary_e(mask, BinaryOps.MUL, grad_output)

# Step 18 - Log
class Log(Function):
    def forward(self, x):
        # TODO: return the natural log of x and save x for backward
        UnaryOps, _, _, _ = make_op_enums()
        self.x = x 
        return x.e(UnaryOps.LOG)

    def backward(self, grad_output):
        # TODO: return the gradient of log with respect to its input
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(grad_output, BinaryOps.DIV, self.x)

# Step 19 - Exp
class Exp(Function):
    def forward(self, x):
        # TODO: compute the elementwise exponential and keep what backward needs
        UnaryOps, _, _, _ = make_op_enums()
        y = x.e(UnaryOps.EXP)
        self.ret = y 
        return self.ret 

    def backward(self, grad_output):
        # TODO: turn the upstream gradient into the gradient w.r.t. the input
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(self.ret, BinaryOps.MUL, grad_output)

# Step 20 - Sqrt
class Sqrt(Function):
    def forward(self, x):
        # TODO: compute the elementwise square root and cache it for backward
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.SQRT)
        return self.ret 

    def backward(self, grad_output):
        _, BinaryOps, _, _ = make_op_enums()
        two = LazyBuffer.const(2, self.ret.shape)
        denom = lazybuffer_binary_e(two, BinaryOps.MUL, self.ret)
        return lazybuffer_binary_e(grad_output, BinaryOps.DIV, denom)

# Step 21 - Sigmoid
class Sigmoid(Function):
    def forward(self, x):
        # TODO: return the elementwise logistic activation of LazyBuffer x
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.SIGMOID)
        return self.ret

    def backward(self, grad_output):
        # TODO: return grad_output times the sigmoid derivative
        _, BinaryOps, _, _ = make_op_enums()
        ones = LazyBuffer.const(1, self.ret.shape)
        neg = lazybuffer_binary_e(ones, BinaryOps.SUB, self.ret)
        inter = lazybuffer_binary_e(self.ret, BinaryOps.MUL, neg)
        return lazybuffer_binary_e(inter, BinaryOps.MUL, grad_output)

# Step 22 - Add
class Add(Function):
    def forward(self, x, y):
        # TODO: return the elementwise sum of LazyBuffers x and y
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(x, BinaryOps.ADD, y)

    def backward(self, grad_output):
        # TODO: route grad_output to each input that requires a gradient
        dx = grad_output if self.needs_input_grad[0] else None
        dy = grad_output if self.needs_input_grad[1] else None
        return dx, dy

# Step 23 - Sub
class Sub(Function):
    def forward(self, x, y):
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(x, BinaryOps.SUB, y)

    def backward(self, grad_output):
        UnaryOps, _, _, _ = make_op_enums()

        dx = grad_output if self.needs_input_grad[0] else None
        dy = grad_output.e(UnaryOps.NEG) if self.needs_input_grad[1] else None

        return dx, dy

# Step 24 - Mul
class Mul(Function):
    def forward(self, x, y):
        _, BinaryOps, _, _ = make_op_enums()
        self.x = x
        self.y = y
        return lazybuffer_binary_e(x, BinaryOps.MUL, y)

    def backward(self, grad_output):
        _, BinaryOps, _, _ = make_op_enums()

        grad_x = (
            lazybuffer_binary_e(grad_output, BinaryOps.MUL, self.y)
            if self.needs_input_grad[0] else None
        )

        grad_y = (
            lazybuffer_binary_e(grad_output, BinaryOps.MUL, self.x)
            if self.needs_input_grad[1] else None
        )

        return grad_x, grad_y

# Step 25 - Div
class Div(Function):
    def forward(self, x, y):
        _, BinaryOps, _, _ = make_op_enums()
        self.x = x
        self.y = y
        return lazybuffer_binary_e(x, BinaryOps.DIV, y)

    def backward(self, grad_output):
        UnaryOps, BinaryOps, _, _ = make_op_enums()

        grad_x = grad_y = None

        if self.needs_input_grad[0]:
            grad_x = lazybuffer_binary_e(grad_output, BinaryOps.DIV, self.y)

        if self.needs_input_grad[1]:
            y_sq = lazybuffer_binary_e(self.y, BinaryOps.MUL, self.y)
            num = lazybuffer_binary_e(grad_output, BinaryOps.MUL, self.x)
            grad_y = lazybuffer_binary_e(num, BinaryOps.DIV, y_sq)
            grad_y = grad_y.e(UnaryOps.NEG)

        return grad_x, grad_y

# Step 26 - sum_function_forward
class Sum(Function):
    def forward(self, x, axis):
        # TODO: Reduce x with ReduceOps.SUM over axis (keepdims) and cache shape/axis.
        self.input_shape = x.shape 
        self.axis = axis 
        result = x._np.sum(axis = axis, keepdims=True)
        return LazyBuffer(result)

# Step 27 - sum_function_backward
def backward(self, grad_output):
    # TODO: broadcast the summed gradient back to the original input shape
    grad = grad_output._np 
    grad = np.broadcast_to(grad, self.input_shape)
    grad = np.ascontiguousarray(grad)
    return LazyBuffer(grad)

# Step 28 - max_function_forward
class Max(Function):
    def forward(self, x, axis):
        # TODO: reduce x with the MAX reduce op along axis and cache for backward
        self.x = x
        self.axis = axis 
        arr = self.x._np.max(axis = axis, keepdims=True)
        self.ret = LazyBuffer(arr)
        return self.ret

# Step 29 - max_function_backward
LazyBuffer.r = r
LazyBuffer.reshape = reshape
LazyBuffer.expand = expand
LazyBuffer.permute = permute

def backward(self, grad_output):
    _, BinaryOps, ReduceOps, _ = make_op_enums()

    x_shape = self.x._np.shape

    # Broadcast the reduced max back to the input shape
    ret_exp = self.ret.expand(x_shape)

    # Build a mask of maximal elements: 1 - (x < max)
    ones = LazyBuffer.const(1, x_shape)
    lt = lazybuffer_binary_e(self.x, BinaryOps.CMPLT, ret_exp)
    mask = lazybuffer_binary_e(ones, BinaryOps.SUB, lt)

    # Count ties and broadcast the counts
    counts = mask.r(ReduceOps.SUM, self.axis)
    counts = counts.expand(x_shape)

    # Split the gradient evenly among tied maxima
    weights = lazybuffer_binary_e(mask, BinaryOps.DIV, counts)

    # Broadcast upstream gradient
    grad = grad_output.expand(x_shape)

    return lazybuffer_binary_e(weights, BinaryOps.MUL, grad)

Max.backward = backward

# Step 30 - Reshape
class Reshape(Function):
    def forward(self, x, shape):
        # TODO: cache the input shape and return x reshaped to shape
        self.x = x 
        self.input_shape = x._np.shape 
        self.ret = self.x._np.reshape(shape)
        return LazyBuffer(self.ret)


    def backward(self, grad_output):
        # TODO: reshape the gradient back to the cached input shape
        return grad_output.reshape(self.input_shape)

# Step 31 - expand_function_forward
def expand_function_forward(ctx, x, shape):
    # TODO: cache x.shape on ctx, then broadcast x to the target shape
    ctx.input_shape = x.shape 
    new_shape = tuple(int(d) for d in shape)
    out = np.broadcast_to(x._np, new_shape)
    return LazyBuffer(np.ascontiguousarray(out))

# Step 32 - expand_function_backward
def expand_function_backward(ctx, grad_output):
    input_shape = ctx.input_shape
    grad_shape = grad_output.shape

    axes = tuple(i for i in range(len(input_shape))
                 if input_shape[i] == 1 and grad_shape[i] != 1)

    result = grad_output.r(SimpleNamespace(name='SUM'), axes)
    return result.reshape(input_shape)

# Step 33 - permute_function_forward_backward
def permute_function_forward_backward():
    # TODO: return (forward, backward); forward reorders axes, backward inverts the order
    def forward(ctx, x, order): 
        ctx.order = order 
        return permute(x, order)
    def backward(ctx, grad_output): 
        inv = argsort(ctx.order)
        return permute(grad_output, inv)
    return forward, backward

# Step 34 - Tensor
class Tensor:
    def __init__(self, data, requires_grad=False, _ctx = None):
        # TODO: wrap data in a LazyBuffer and store grad/ctx bookkeeping
        if isinstance(data, LazyBuffer):
            self.lazydata = data
        else:
            arr = np.asarray(data, dtype=np.float32)
            self.lazydata = LazyBuffer(arr)

        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = _ctx

    @property
    def data(self):
        # TODO: return the underlying LazyBuffer
        return self.lazydata

    @data.setter
    def data(self, value):
        # TODO: replace the underlying LazyBuffer
        self.lazydata = value

    @property
    def shape(self):
        return self.lazydata.shape

    @property
    def dtype(self):
        return self.lazydata.dtype

    def numpy(self):
        return self.lazydata._np

# Step 35 - tensor_from_data
def tensor_from_data(data, requires_grad=False):
    # TODO: wrap a number, list, or numpy array in a LazyBuffer held by a Tensor
    if not isinstance(data, LazyBuffer): 
        data = np.asarray(data, dtype = np.float32)
        data = LazyBuffer(data)
    return Tensor(data, requires_grad=requires_grad)

# Step 36 - tensor_creation_helpers
def tensor_creation_helpers():
    # TODO: return (zeros_fn, ones_fn, full_fn) building constant-filled Tensors
    def full_fn(shape, value): 
        arr = np.full(shape, value, dtype=np.float32)
        return Tensor(LazyBuffer(arr))
    def zeros_fn(shape): 
        return full_fn(shape, 0.0)
    def ones_fn(shape): 
        return full_fn(shape, 1.0)
    return(zeros_fn, ones_fn, full_fn)

# Step 37 - tensor_randn
def tensor_randn(shape, seed=None, requires_grad=False):
    # TODO: Create a Tensor of standard-normal samples for the given shape.
    shape = tuple(int(i) for i in shape)
    seeded_rng = np.random.RandomState(seed)
    u = seeded_rng.random((2,) + shape)
    u1 = u[0]
    u2 = u[1]
    u1_clipped = np.clip(u1, 1e-12, 1.0)
    # Apply the Box-Muller formula, cast to float32, wrap in LazyBuffer, then in Tensor with requires_grad.
    # Review this once more
    z = np.sqrt(-2.0 * np.log(u1_clipped)) * np.cos(2.0 * np.pi * u2)
    z = z.astype(np.float32)

    return Tensor(LazyBuffer(z), requires_grad=requires_grad)

# Step 38 - build_topological_order
def build_topological_order(tensor):
    order = []
    visited = set()

    def dfs(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        if node._ctx is not None:
            for p in node._ctx.parents:
                dfs(p)
        order.append(node)

    dfs(tensor)
    return order

# Step 39 - tensor_backward
def tensor_backward(tensor):
    # 1. Seed the output gradient
    tensor.grad = Tensor(LazyBuffer(np.ones(tensor.shape, dtype=np.float32)))

    # 2. Topological order, reversed: output first, leaves last
    order = build_topological_order(tensor)

    for node in reversed(order):
        if node._ctx is None or node.grad is None:
            continue

        # 3. Backward expects/returns LazyBuffers, so unwrap node.grad.data
        grads = node._ctx.backward(node.grad.data)

        # 4. Normalize single-return ops (unary) to a tuple
        if isinstance(grads, LazyBuffer):
            grads = (grads,)

        # 5. Route each gradient to its parent
        for parent, g in zip(node._ctx.parents, grads):
            if g is None or not parent.requires_grad:
                continue
            if parent.grad is None:
                parent.grad = Tensor(g)
            else:
                parent.grad = Tensor(LazyBuffer(parent.grad.data._np + g._np))

    return None

# Step 40 - bind_unary_tensor_methods
def function_apply(FunctionCls, *tensors, **kwargs):
    return FunctionCls.apply(*tensors, **kwargs)
    
def bind_unary_tensor_methods():
    def _make(F):
        def method(t):
            return function_apply(F, t)
        return method

    methods = {
        'neg':     _make(Neg),
        'relu':    _make(Relu),
        'log':     _make(Log),
        'exp':     _make(Exp),
        'sqrt':    _make(Sqrt),
        'sigmoid': _make(Sigmoid),
    }
    return methods

# Step 41 - broadcasted
def broadcasted(x, y):
    rawX = x.data._np
    rawY = y.data._np

    bx, by = np.broadcast_arrays(rawX, rawY)

    common_shape = bx.shape

    if x.shape == common_shape:
        new_x = x
    else:
        new_x = tensor_from_data(np.array(bx, dtype=np.float32))

    if y.shape == common_shape:
        new_y = y
    else:
        new_y = tensor_from_data(np.array(by, dtype=np.float32))

    return new_x, new_y

# Step 42 - bind_binary_tensor_methods
def bind_binary_tensor_methods():
    # TODO: attach broadcasting add/sub/mul/div methods onto the Tensor class
    # Define four small closures, one per op 
    def add(self,other): 
        x, y = broadcasted(self, other)
        return Add.apply(x, y)
    Tensor.add = add 
    Tensor.__add__ = add 
    
    def mul(self,other): 
        x, y = broadcasted(self, other)
        return Mul.apply(x, y)
    Tensor.mul = mul 
    Tensor.__mul__ = mul 

    def sub(self,other): 
        x, y = broadcasted(self, other)
        return Sub.apply(x, y)
    Tensor.sub = sub 
    Tensor.__sub__ = sub  

    def div(self,other): 
        x, y = broadcasted(self, other)
        return Div.apply(x, y)
    Tensor.div = div 
    Tensor.__div__ = div

# Step 43 - bind_movement_tensor_methods
def bind_movement_tensor_methods():
    """
        Some Theory: 
        1.  Building classes at runtime: 
            type(name, bases, namespace)
                name -- a string 
                bases -- a tuple of parent classes 
                namespace -- a dict that becomes the classes' attributes and methods 
            We build the class dynamically at runtime.
        2. Tensor.__new__(Tensor)  
            Allocates a blank, empty instance in memory (no attributes set yet)
            **Why This??** Because we do not want the extra work done here, helps us not do redundant work and lets us build the lightest possible wrapper
        3. Tensor.__init__(instance, data)
            runs your __init__ body on that blank instance setting self.lazydata, self.requires_grad, etc... 
    """
    Expand = type(
        'Expand', 
        (Function, ), {
            'forward': expand_function_forward, 
            'backward': expand_function_backward, 
        }
    )
    _permute_fwd, _permute_bwd = permute_function_forward_backward()
    Permute = type(
        'Permute', 
        (Function, ), {
            'forward': _permute_fwd, 
            'backward': _permute_bwd, 
        }
    )

    def _get_lazybuffer(t): 
        for attr in ('lazydata', 'data', '_lazydata'): 
            if hasattr(t, attr): 
                val = getattr(t, attr)
                if isinstance(val, LazyBuffer):
                    return val 
        return LazyBuffer(np.asarray(t))

    def _wrap(out): 
        t = Tensor.__new__(Tensor)
        t.lazydata = out
        return t 

    def reshape(t, shape):
        buf = _get_lazybuffer(t)
        ctx = object.__new__(Reshape)
        out = ctx.forward(buf, shape=tuple(shape))
        return _wrap(out)

    def expand(t, shape):
        buf = _get_lazybuffer(t)
        ctx = object.__new__(Expand)
        out = ctx.forward(buf, shape=tuple(shape))
        return _wrap(out)

    def permute(t, order):
        buf = _get_lazybuffer(t)
        ctx = object.__new__(Permute)
        out = ctx.forward(buf, order=tuple(order))
        return _wrap(out)

    return {'reshape': reshape, 'expand': expand, 'permute': permute}

# Step 44 - bind_reduce_tensor_methods
def bind_reduce_tensor_methods():
    def _axes(ndim, axis):
        if axis is None:
            return tuple(range(ndim))
        if isinstance(axis, int):
            return (axis % ndim,)
        # tuple/list of axes
        return tuple(a % ndim for a in axis)

    def _np(self):
        for attr in ('_np', 'lazydata', 'data', 'buffer'):
            if hasattr(self, attr):
                val = getattr(self, attr)
                if isinstance(val, np.ndarray):
                    return val
                if isinstance(val, LazyBuffer):
                    return val._np
        return np.asarray(self)

    def sum(self, axis=None, keepdim=False):
        arr = _np(self)
        axes = _axes(arr.ndim, axis)
        result = arr.sum(axis=axes, keepdims=keepdim)
        return tensor_from_data(result.tolist())

    def max(self, axis=None, keepdim=False):
        arr = _np(self)
        axes = _axes(arr.ndim, axis)
        result = arr.max(axis=axes, keepdims=keepdim)
        return tensor_from_data(result.tolist())

    Tensor.sum = sum
    Tensor.max = max
    if not hasattr(Tensor, 'numpy'):
        Tensor.numpy = lambda self: _np(self)

    return {'sum': sum, 'max': max}

# Step 45 - tensor_mean
def tensor_mean(x, axis=None, keepdim=False):
    def _np(t):
        for attr in ('lazydata', 'data', '_lazydata', 'buffer'):
            if hasattr(t, attr):
                buf = getattr(t, attr)
                return buf._np if hasattr(buf, '_np') else buf
        return np.asarray(t)

    arr = _np(x)
    outp = arr.mean(axis=axis, keepdims=keepdim)
    return tensor_from_data(outp.tolist() if hasattr(outp, 'tolist') else outp)

# Step 46 - tensor_transpose
def tensor_transpose(x, ax1=-2, ax2=-1):
    # TODO: swap axes ax1 and ax2 of tensor x using a permutation
    buf = None
    for attr in ('lazydata', 'data', '_lazydata', 'buffer'): 
        if hasattr(x, attr): 
            buf = getattr(x, attr)
            break 
    arr = buf._np if hasattr(buf, '_np') else buf 
    # Normalize the axes and get rank
    n = len(arr.shape)
    a1 = ax1 % n 
    a2 = ax2 % n 
    # Build the swapped order 
    order = list(range(n))
    order[a1], order[a2] = order[a2], order[a1]
    return x.permute(order)

# Step 47 - tensor_matmul_2d
def tensor_matmul_2d(a, b):
    # Pull the underlying numpy arrays out of a and b
    def _np(t):
        for attr in ('lazydata', 'data', '_lazydata', 'buffer', '_data'):
            if hasattr(t, attr):
                buf = getattr(t, attr)
                return buf._np if hasattr(buf, '_np') else buf
        return np.asarray(t)

    an = _np(a)
    bn = _np(b)

    # 1. Shapes
    m, k = an.shape
    _, n = bn.shape

    # 2. Reshape to broadcast-compatible 3D shapes
    a3 = an.reshape((m, k, 1))
    b3 = bn.reshape((1, k, n))

    # 3. Elementwise multiply, broadcasting to (m, k, n)
    prod = a3 * b3

    # 4. Reduce over the shared k axis
    result = prod.sum(axis=1)  # shape (m, n)

    # 5. Wrap back into a Tensor of the same class as a
    return type(a)(result.tolist())

# Step 48 - tensor_softmax
def tensor_softmax(x, axis=-1):
    def _np(t):
        for attr in ('lazydata', 'data', '_lazydata', 'buffer', '_data'):
            if hasattr(t, attr):
                buf = getattr(t, attr)
                return buf._np if hasattr(buf, '_np') else buf
        return np.asarray(t)

    xn = _np(x)
    m = xn.max(axis=axis, keepdims=True)
    e = np.exp(xn - m)
    denom = e.sum(axis=axis, keepdims=True)
    out = e / denom
    return tensor_from_data(out.tolist())

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

