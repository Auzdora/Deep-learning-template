from trainer import *


class A():
    def __init__(self,animal, name, age):
        self.animal = animal
        self.name = name
        self.age = age

    def grow(self):
        print(f'2 years later,{self.name} age is becoming {self.age+2}'.format(self=self))


class B(A):
    def __init__(self,animal,name,age,sex):
        self.sex = sex
        self.init_kwargs = {
            'animal':animal,
            'name':name,
            'age':age
        }
        super().__init__(**self.init_kwargs)


lily = B('mankind','Lily',15,'gril')
lily.grow()