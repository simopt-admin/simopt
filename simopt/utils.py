class ClassPropertyDescriptor:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner):
        return self.fget(owner)


def classproperty(func):
    return ClassPropertyDescriptor(func)
