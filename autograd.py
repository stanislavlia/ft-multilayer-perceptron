import math


class Value:
    def __init__(self, val, _parents=(), _op='', _label='value'):
        
        self.val = val
        self._prev = set(_parents)
        self._op = _op
        self.grad = 0.0
        self._label = _label
        self._backward = lambda: None

    

    def __add__(self, another_value):
        #Make sure we manipulate with Value instances
        another_value = another_value if isinstance(another_value, Value) else Value(another_value)
        
        #We need to accumulate gradients
        #when this node is used multiple times
        result = self.val + another_value.val
        out = Value(result, _parents=(self, another_value), _op='+')
        
        #Define function for backprop for sum
        def _backward():
            
            self.grad += out.grad
            another_value.grad += out.grad
        
        out._backward = _backward
        
        
        return out
    
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, another_value):
        return self + (-another_value)
    
    
    def __truediv__(self, another_value):
        return self * (another_value ** -1)
    
    def __mul__(self, another_value):
        #Make sure we manipulate with Value instances
        another_value = another_value if isinstance(another_value, Value) else Value(another_value)
                
        result = self.val * another_value.val
        out = Value(result, _parents=(self, another_value), _op='*')
        
        def _backward():
            self.grad += another_value.val * out.grad
            another_value.grad += self.val * out.grad
            
        out._backward = _backward
    
        return out
    
    def __rmul__(self, another_value):
        return self * another_value
    
    def __radd__(self, another_value):
        return self + another_value
    
    def __rsub__(self, another_value):
        
        another_value = another_value if isinstance(another_value, Value) else Value(another_value)
        
        return another_value + (-self)
    
    
    def tanh(self):
        """Activation function: TanH"""

        val = (math.exp(2 * self.val) - 1) / (math.exp(2 * self.val) + 1)
        
        out = Value(val, _parents=(self, ), _op='tanh')
        
        def _backward():
            self.grad +=  (1 - out.val ** 2) * out.grad
            
        out._backward = _backward
        
        return out
    
    def relu(self):
        """Activation function: RELU"""

        val = max(self.val, 0)
        out = Value(val, _parents=(self,) , _op='relu')

        def _backward():
            self.grad += int(out.val > 0) * out.grad
        
        out._backward =_backward

        return out
    
    def sigmoid(self):
        """Activation function: Sigmoid"""

        val = 1 / ( 1 + math.exp(-self.val))
        out = Value(val, _parents = (self, ), _op="sigmoid")

        def _backward():
            self.grad += ((1 - out.val) * out.val ) * out.grad

        out._backward = _backward

        return out

    
    def __pow__(self, another_value):
        assert isinstance(another_value, (int, float)), "only int or float expected"
        
        out = Value(self.val ** another_value, _parents=(self, ), _op=f"^{another_value}")
        
        def _backward():
            self.grad += (another_value * self.val ** (another_value - 1)) * out.grad
            
        out._backward = _backward
        
        
        return out
        
    
    
    def exp(self):
        
        result = math.exp(self.val)
        out = Value(result, _parents=(self,), _op="exp")
        
        def _backward():
            self.grad += result * out.grad
            
        out._backward = _backward
        return out
    
    
    def backward(self):
        
        self.grad = 1
        
        topo = []
        visited = set()

        #Topological sorting
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
         
        build_topo(self)
        
        #go through node and call _backward()
        for node in reversed(topo):
            node._backward()
            
    def __repr__(self):
        return f"Value(val={self.val}, label={self._label})"


def test_autograd():
    x = Value(2)
    y = Value(0.7)

    print("=========TEST AUTOGRADIENT======")

    #Compute your function step by  step
    print(x)
    print(y)
    step1 = (x - 0.2) ** 3
    step2 = y.exp() ** -2
    step3 = step1 + step2 + x*y


    #run backpropogation on last step
    step3.backward()

    print(f"gradient of x wrt to F is {x.grad}")  #dF/dx =  10.42
    print(f"gradient of y wrt to F is {y.grad}")  #dF/dy = 1.506

if __name__ == "__main__":
    test_autograd()
