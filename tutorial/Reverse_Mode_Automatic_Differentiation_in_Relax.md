# Reverse Mode Automatic Differentiation in Relax

*Notice: The syntax of the Relax language used in this document  may be outdate since current parser is not stable.*

Implementation Details of the experimental `Gradient` Pass in Relax.

Updated: 2023/2/7

# 1. <span id="c1">Code and API</span>

## Code Organization

The pass is in `src/relax/transform/gradient.cc` , with the Python API in `python/relax/transform/transform.py`.

The gradients of operators are registered in `python/relax/op/_op_gradient.py`.

## Transformation

The pass differentiates the input relax function and return a fresh new differentiated function, with the name `<original_name>_adjoint`. It returns both the original return value and the needed adjoints in the form of Tuple. See the following example:

```python
@I.ir_module
class Before:
    @R.function
    def main(
        x: R.Tensor((5, 5), dtype="float32"), y: R.Tensor((5, 5), dtype="float32")
    ) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv0: R.Tensor((5, 5), dtype="float32") = R.add(x, y)
            gv0: R.Tensor((), dtype="float32") = R.sum(lv0, axis=None, keepdims=False)
            R.output(gv0)
        return gv0

@I.ir_module
class After:
    @R.function
    def main(
        x: R.Tensor((5, 5), dtype="float32"), y: R.Tensor((5, 5), dtype="float32")
    ) -> R.Tensor((), dtype="float32"):
        with R.dataflow():
            lv0: R.Tensor((5, 5), dtype="float32") = R.add(x, y)
            gv0: R.Tensor((), dtype="float32") = R.sum(lv0, axis=None, keepdims=False)
            R.output(gv0)
        return gv0

    @R.function
    def main_adjoint(
        x: R.Tensor((5, 5), dtype="float32"), y: R.Tensor((5, 5), dtype="float32")
    ) -> R.Tuple(
        R.Tensor((), dtype="float32"),
        R.Tuple(R.Tensor((5, 5), dtype="float32"), R.Tensor((5, 5), dtype="float32")),
    ):
        with R.dataflow():
            lv0: R.Tensor((5, 5), dtype="float32") = R.add(x, y)
            gv0: R.Tensor((), dtype="float32") = R.sum(lv0, axis=None, keepdims=False)
            gv0_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
            lv0_adjoint: R.Tensor((5, 5), dtype="float32") = R.broadcast_to(
                gv0_adjoint, (5, 5)
            )
            x_adjoint: R.Tensor((5, 5), dtype="float32") = lv0_adjoint
            y_adjoint: R.Tensor((5, 5), dtype="float32") = lv0_adjoint
            R.output(gv0, x_adjoint, y_adjoint)
        return (gv0, (x_adjoint, y_adjoint))
```

There are some important design points of this AD Pass.

- It is IRModule to IRModule. The target function `main` is preserved in new module and new differentiated function `main_adjoint` is added to the module.
- The `main_adjoint` contains the original part (forward part) of `main` and appends new adjoint bindings after.
- `main_adjoint` returns a pair `(orig_ret, grads)`. `orig_ret` is the return value of `main`. And `grads` is the adjoints of specified arguments. For each argument of the original call in `require_grads`, the pass stores its partial in the corresponding position (in order).



## Restrictions

### R1. Currently AD can only work in:

- A relax function with a **single dataflow-block**.
- The function's body should be `SeqExpr`
- The return value should be a `Scalar`.

### R2. Currently considered AST nodes:

- [x] Primitive Call
- [x] Assignment
- [x] Tuple-aware (Tuple, TupleGetItem)
- [x] Constant
- [ ] Match Shape
- [ ] Call TIR

### R3. AD uses the following built-in operators:

- `relax.zeros`
- `relax.ones`
- `relax.add`

### R4. Gradients of Operators

The gradients of operators presented in the module must be registered properly. See [2. Gradients of Operators](#c2) for details.



## API

```c++
TVM_DLL Pass Gradient(GlobalVar global_var, Optional<Array<Var>> require_grads);
```

- `global_var` is the GlobalVar of the specific function.
- `require_grads` specifies the relax variables whose adjoints are needed. Must be parameters of the given function. If it is not specified, adjoints of all arguments would be computed.

- Returns a new module with the original function and the new differentiated function.



# 2. <span id="c2">Gradients of Operators</span>

The gradient is registered as an attribute of the operator, which `attr_key` is `FPrimalGradient`. The type of the gradient is

```c++
using FPrimalGradient = runtime::TypedPackedFunc<tvm::Array<Expr>(
    const Var& orig_var, const Call& orig_call, const Var& output_grad, const BlockBuilder& ctx)>;
```

For example, the gradient of `relax.tanh` can be written as

```python
@register_gradient("relax.tanh")
def tanh_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of tanh.

    Forward Form:
        y = relax.tanh(x)

    Backward:
        Returns [y_grad * (1 - y * y)].
    """
    x_ones = ones(_get_shape(orig_call.args[0]), orig_call.args[0].struct_info.dtype)
    return [
        multiply(
            output_grad,
            subtract(x_ones, multiply(orig_var, orig_var)),
        )
    ]
```

Some illustrations about the gradient registering.

### I1. Arguments of the gradient function

Currently we have four arguments

```
orig_var: Var, orig_call: Call, output_grad: Var, ctx: BlockBuilder
```

```
y = relax.tanh(x)
```

- `orig_call` is the orginal call expr which we want to differentiate, which is `Call(relax.tanh, [x,], ...)`. The inputs can be fetched by this Call expr,

- `output_grad` is the gradient of RHS, which is supposed to be `y_adjoint`.

- `orig_var` is `y`. It is passed to saving some calculations. In this example, using `y` can prevent us from recalculating `tanh(x)`.

- `ctx` is the context which is not used right now. But we believe it is useful when it comes to dynamic shape cases and when we need to emit some bindings or do some normalizations.

### I2. Collapse_Sum

Because broadcasting happens in the forward calculations, we need an opposite operator of `broadcast` when we go backward. In tvm codebase we call this operator `collapse_sum` and it is important in AD. And we use it when differentiating these broadcasting operators.

If we have enough information to prove shape equality, we can eliminate some `collapse_sum` in the gradient function.

```
c = relax.add(a, b)
a_adjoint = relax.collapse_sum_to(c_adjoint, a.shape)
# If we can prove a.shape == c_adjoint.shape, this collapse_sum_to is unnecessary.
```



### I3. Some problems

Here are some problems which are not handled well now. Some of this problems relate to the limitations of `topi` and current relax supports.

-  Dynamic Shape / Shape Var
- Complicated NN Operators



# 3. <span id="c3">Simplest Case: AD with Only Primitive Operators</span>

In this simplest case, we only consider two types of bindings: Call and Assignment. And we can give the first version AD

### V1: Simplest Case AD

```python
def AD():
    for binding in reverse_order:
        adjoint_var = Var()
        adjoint_var_map_[binding->var] = adjoint_var

        if binding->var not in adjoint_expr_map_: # irrelevant parts or target
            if binding->var == target:
                adjoint_expr_map_[binding->var] = 1 # target
            else:
                continue # irrelevant parts

        adjoint_expr = adjoint_expr_map_[binding->var]
        emit_and_bind(adjoint_var, adjoint_expr) # emit and bind

        call = cast<Call>(binding->value) # in this case value can only be call
        gradient = gradient_map[call->op]
        partials = gradient(call, adjoint_var)
        # AD. Note that we pass the adjoint_var to it instead of adjoint_expr !!!

    for i in len(call.args):
        arg = call.args[i]
        partial = partials[i]
        update_adjoint(arg, partial)

def update_adjoint(arg, partial)
    if arg not in adjoint_expr_map_:
    	adjoint_expr_map_[arg] = partial # frist update
    else:
    	adjoint_expr_map_[arg] += partial

# It is clear that after the for, we have adjoint_expr for each input.
# Then we just need to bind them with adjoint_var and emit them.
```

There are some points about the first version AD.

### P1. Expr and Var of Adjoints

In Relax, like many other programming languages, variables and expressions (or values) are separated. And if we don't handle this issue well, unnecessary calculations appear in the transformed module.

The correct way we implement our AD should be:

- Allocate Vars for every adjoints. For a Var `a` in forward, we allocate a new Var `a_adjoint` as its adjoint. To separate Expr and Var, we can let `a_adjoint_var` be this var and `a_adjoint_expr` be the corresponding calculated Expr.
- Calculate the value of adjoints in Expr using these adjoints in the form of Vars. For example, if a propagation rule is `c_adjoint = a_adjoint + b_adjoint`, we should write as `c_adjoint_expr = Call(add, [a_adjoint_var, b_adjoint_var])` .
- Bind and emit every pair of Var and Expr.

In practice, we need two Maps: one for Var and another for Expr.

```c++
// var to its adjoints var
Map<Var, Var> adjoint_var_map_;
// var to its adjoint expr
Map<Var, Expr> adjoint_expr_map_;
```

Indeed there are two styles of implementations in respect of `adjoint_expr_map_`. Another way is using a `Map<Var, Array<Expr>> adjoint_expr_map_;` , with the idea that store the partials first and add them together finally. Here we adopt the first way, `Map<Var, Expr>`. And each time doing update for the adjoint of variable v, we can just use the logic like this (Pseudo Code)

```c++
adjoint_expr_map_.Set(v, adjoint_expr_map_.Get(v) + increment)
```
As stated before, if it is the first time we update to v, `+=` becomes `=` :

```
adjoint_expr_map_.Set(v, increment)
```

### P2. Assignment

```
b = a
```

"Assignment" is not a call, but it can be viewed as an "identity" operator. The logic to handle this case is similar.

### P3. About Irrelevant Parts

```python
@I.ir_module
class Before:
    @R.function
    def main(x: Tensor((5, 5), "float32"),
            y: Tensor((5, 5), "float32")):
        with R.dataflow():
            lv0 = relax.add(x, y)
            lv1 = relax.sub(x, y)
            lv2 = relax.sum(lv1)
            gv0 = relax.sum(lv0)
            R.output(gv0)
        return gv0
```

In this example, `lv1`, `lv2` has no contributions to `gv0`. So mathematically
$$
\frac{\partial\text{gv0}}{\partial \text{lv1}} = \frac{\partial\text{gv0}}{\partial \text{lv2}} = 0
$$
which means  `lv1_adjoint` and `lv2_adjoint` has no contributions to `x`, `y` and any other parts. So for these irrelevant parts, we can safely just ignore them when doing AD. "Ignore" means when we visit these bindings,

```
lv1 = relax.sub(x, y)
lv2 = relax.sum(lv1)
```

we can just do nothing and return.



### P4. Some Interesting Views

We can conclude some interesting and reasonable rules for AD. First, in a horizontal view, focus on a single binding

```
d = func(a, b, c)
```

While the direction original (forward) dataflow is `a, b, c -> d`, AD reverses everything and makes the adjoint dataflow be `d_adjoint->a_adjoint, b_adjoint, c_adjoint`, which is from the **user** `d` and to **uses** `a` `b` `c`.

Moreover, in a vertical view, focus on a single variable across bindings

```
def of a
...
use1 of a
use2 of a
use3 of a
```

In the reverse-mode,

```
use3 of a
use2 of a
use1 of a
...
def of a
```

In each use, `adjoint_expr_map_[a]` is updated. And when we meet the def, `adjoint_expr_map_[a]` must be updated completely and it is used to update other adjoints. If Relax is SSA, we can say that after we meet the def, the live time of `a_adjoint` is ended.

If there is a def but not used, it must have no contribution to target so (in "About Irrelevant Parts") it is reasonable to ignore them.  If there is a use but no def is found, it is a error in original program.



# 4. <span id="c4">Tuple-Aware AD</span>

In this case we need to consider not only Tensor but also Tuple. Two new types of bindings are introduced:

- relax.Tuple (Tuple Definition): `c = (a, b)`
- relax.TupleGetItem:  `b = a[0]`

Before we start to handle Tuple in AD, there are some important facts about Tuple in Relax:

### F1: Leaf Nodes

In Relax specification, `Tuple` is a leaf but `TupleGetItem` is not. Therefore after normalization, we can have nested Tuple definition but not nested TupleGetItem.

```
c = ((a, b), (e,)) # OK
c = b[1][2]
```

### F2: Struct Info Relation

The adjoint of a Tuple has **exactly the same struct info** (both the nested structure and the leaf sinfo of the tuple info) with the original Tuple. According to this relation, we can conclude a basic idea of tuple-aware AD as:

- Tuple Definition

```python
Before:
c = (a, b)

After:
a_adjoint += c_adjoint[0]
b_adjoint += c_adjoint[1]
```

- TupleGetItem


```python
Before:
b = a[0]

After:
a_adjoint[0] += b_adjoint
```

But if this AD is tuple-aware, then everything can be Tuple. For instance, in the above example, we can ensure some of them are Tuples but  `b_adjoint` can also be a Tuple. And every variable we meet can also be bound to a Tuple. So every operation (initialization, add, ...) should be tuple-aware. Luckily this can be solved elegantly by the nested_msg utils.

The basic idea using nested_msg is that changing our previous `Map<Var, Expr> adjoint_expr_map_` to `Map<Var, NestedMsg<Expr>> adjoint_msg_map_`. And in the internal computing of AD, all elements we face are `NestedMsg<Expr>` thus we will not forget to handle tuple logic in some parts.

Different Tuple operations need different NestMsg uitls. Here are some mainly logic appears in AD which should use NestedMsg.



### N1: Addition

Addition is common in AD because every time we sum the partials up to get the final adjoint. From `Expr` to `Nested<Expr>`, we can't simply call `relax.add` since it only works for Tensor. But we can preserve it as the leaf operation, and use `CombineNestedMsg` to add two `Nested<Expr>`.

- NestedMsg util: `CombineNestedMsg`, which recursively combine two nested message into one.
- `fcombine`: Tensor add (Call `relax.add`).



### N2: Deal With Tuple Definition

```
Before:
c = (a, b)

After:
a_adjoint += c_adjoint[0]
b_adjoint += c_adjoint[1]
```

**Q1:** Where does `c_adjoint` comes from? It is a tuple so where is this tuple created?

**A1**:An adjoint is created when it is firstly updated.

- As for the tensor case, when it is a first update, we replace `+=` by `=` and this skips the process of creation.

- But when it comes to Tuple, we can not just do these. `c_adjoint` is updated only when `c` is used.

```python
Before:
c = (a, b)
d = c[0]
e = c[1]

After:
c_adjoint[1] += e_adjoint # the first update!
c_adjoint[0] += d_adjoint
a_adjoint += c_adjoint[0]
b_adjoint += c_adjoint[1]
```

Then we can see the problem: the update for a Tuple adjoint is a "partial update", which needs `c_adjoint[1]` to be a left value. But this is not supported in Relax semantic. Instead, we must create a "zeros Tuple" skeleton ahead and then do partial updates.

```python
# Invalid: c_adjoint[i] can not be a leaf value
c_adjoint[1] += e_adjoint
c_adjoint[0] += d_adjoint

# Valid: Build a "zeros Tuple" first, and use tuple-aware addition.
c_adjoint = (0 + e_adjoint, 0 + d_adjoint)
```

This "build zeros Tuple" method can be done by `MapToNestedMsg` util.

- NestedMsg util: `MapToNestedMsg`, which maps struct info with possible nested-sinfo to nested message.
- `fmapleaf`:  Call `relax.zeros`.



**Q2:** What if the tuple is not updated completely after all partial updates?

**A2:** This means these postions in Tuple are not used. We can just ignore them. (Letting them be zeros is right!)

After answering these two questions, we can now handle Tuple definition case well. Given that Tuple is a leaf node, it may present in the arguments of a Call. So we should update V1 `update_adjoint`  as a tuple-aware one, which can be simply done by `DecomposeNestedMsg`.

```python
# t = (a, b), a, b are Tensors.
update_adjoint(t, partial)
# will call update_adjoint(t[0], partial) and update_adjoint(t[1], partial)
```

- NestedMsg util: `DecomposeNestedMsg`, which recursively decompose the tuple structure in expr and msg along with it.
- `fcombine`: Tensor update in V1.



### N3:  Deal With TupleGetItem

```
Before:
b = a[0]

After:
a_adjoint[0] += b_adjoint
```

We should consider how to implement this partial update. The difficulty is in Relax `a_adjoint[0]` can not be a left-value. For a var binding, the left value should always be a variable. So we should do this manually by diving into the tuple. That is to say, we need implement a method `AddInTuple` which can do addition in a specific position of Tuple.

Recall that we use `NestedMsg<Expr>` to represent the adjoint expr, the implement should be like

```cpp
NestedMsg<Expr> AddInAdjointMsg(NestedMsg<Expr> adjoint, int pos, NestedMsg<Expr> increment) {
	Array<AdjointMsg> arr = adjoint.NestedArray();
    arr.Set(index, TupleAwareAdd(arr[index], increment));
    return NestedMsg<Expr>(arr);
}
```



### N4: Converting between NestedMsg and Tuple

We do adjoints calculation and propagation in the base of `NestedMsg<Expr>` internally. But our input/output is a Relax Module which use relax.Tuple for nested expr. Therefore the conversion is necessary.

#### Tuple Expr to NestedMsg

- NestedMsg util: `MapToNestedMsgBySInfo`. Similar with `MapToNestedMsg`, but using the struct info to decompose the expr.
- `fmapleaf`:  Return the expr directly.

#### NestedMsg to Tuple Expr

- NestedMsg util: `NestedMsgToExpr`, which maps nested message back to the expr.
- `fmapleaf`:  Put the leaf expr in a NestedMsg.



# 5. Constant

For constant the strategy is very simple. Ignoring when meeting them is OK. Note that Constant is also a leaf node in Relax AST, extra checks should be added in `update_adjoint`.
