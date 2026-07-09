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

# Step 25 - Div (not yet solved)
# TODO: implement

# Step 26 - sum_function_forward (not yet solved)
# TODO: implement

# Step 27 - sum_function_backward (not yet solved)
# TODO: implement

# Step 28 - max_function_forward (not yet solved)
# TODO: implement

# Step 29 - max_function_backward (not yet solved)
# TODO: implement

# Step 30 - Reshape (not yet solved)
# TODO: implement

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

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

