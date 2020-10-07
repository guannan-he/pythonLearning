# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# use import method to include other files in this project
import subfunctions

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# while loop

# isStarted = False
# while True:
#     cmd = input("input a command\n").upper()
#     if cmd == "HELP" or cmd == 'H':
#         print("start or s to start")
#         print("stop or p to stop")
#         print("exit or q to exit")
#     elif cmd == "START" or cmd == 'S':
#         if isStarted:
#             print("already started")
#         else:
#             print("started")
#             isStarted = True
#     elif cmd == "STOP" or cmd == 'P':
#         if isStarted:
#             print("stopped")
#             isStarted = False
#         else:
#             print("already stopped")
#     elif cmd == "EXIT" or cmd == 'Q':
#         print("stopped")
#         break
#     else:
#         print("don't understand")
# print("exit input by user")



# for loop

# for item in range(5, 10, 2):  # start, end, gap
#     print(f"price is {item}") # format string


# lists

# names = ["cock", "cock2"]
# print(names[0])
# names[1] = "suck"
# names.append("bit")  # add at end
# names.insert(1, "byte")  # insert at position
# names.remove("bit")  # delete a value
# names.index("cock")  # first index, -1 if not in the list
# names.pop()  # use as a stack
# names.count("cock")  # count
# names.sort()  # sort at ascending order
# names2 = names  # same area in memory
# names3 = names.copy()  # different memory area, get a copy

# tuples
# tuples can't be modified
# use () to define a tuple

# dictionary
# store key-value pair, use ':' to separate

# dict = {
#     "name": "cock",
#     "age": 30,
#     "job": "blow"
#     # keys need be unique
# }
# print(dict["name"])
# # print(dict["doesn't exist"])
# # if use dict["doesn't exist"] will return error
# print(dict.get("doesn't exist"))
# # if use dict.get("doesn't exist") will return "None"
# dict["doesn't exist"] = "now it exists"  # add a key-value pair
# subfunctions.printName("cock")

# try-except block

# try:
#     age = int(input("enter you age:"))
#     a = 1 / age
#     print(a)
# except ZeroDivisionError:
#     print("age can't equals to zero")

# the use of class
# class myClass:
#     x = 0
#     y = 0
#
#     def show(self):
#         print(f"x = {self.x}, y = {self.y}")
#
#     def __init__(self, x, y):  # construction function
#         self.x = x
#         self.y = y
#
#
# myClassObject = myClass(2,10)
# myClassObject.show()


# inherit
# class Mammal:
#     def walk(self):
#         print("walk")
#
# class Dog(Mammal):
#     def bark(self):
#         print("bark")
#
# class Cat(Mammal):
#     def meow(self):
#         print("meow")
#
#
# lulu = Cat()
# lulu.meow()


exit(0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
