from typing import Callable

#This isn't in the LlamaIndex official package, just adding this in. (I think it's elegant)
class composable_strat:
    
    #We can override the RSHIFT operator like in Airflow
    # strategy1 >> strategy2 >> strategy3
    def __init__(self, func: Callable):
        self.func = func

    #A function call to itself returns itself
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    #Right shift overload to compose strategies together
    def __rshift__(self, right):
        return composable_strat(lambda *args, **kwargs: right(*self(*args, **kwargs)))

#Sandbox: Testing if this works
if __name__ == "__main__":
    @composable_strat
    def add1(n: int, a: int) -> (int, int):
        print("add1")
        return n+1, 0

    @composable_strat
    def add2(n, b) -> int:
        print("add2")
        return n+2
    f = add1 >> add2
    print(f(1, 2))


#Levels of understandings
'''
1: Don't know, don't care
2: Heard of it
3: Understand conversation, maybe implement out-of-box (can use)
4: Better implementing out of box, some experimenting (can tell if it's good or bad)
5: Can implement custom features that are better (understands deeply)

'''