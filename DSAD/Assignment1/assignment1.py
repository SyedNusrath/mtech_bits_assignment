import time
from collections import defaultdict
class Node:
    ''' Each node has its data and a pointer that points to next node '''
    def __init__(self, value=None):
        self.value = value
        self.next = None
    
    def __str__(self):
        '''represents the class objects as a string'''
        return str(self.value)


class LinkedList:
    ''' Defining linked list '''
    def __init__(self):
        self.head = None
        self.tail = None
    
class Queue:
    def __init__(self):
        '''Initialize Queue by calling linked list'''
        self.linkedList = LinkedList()
    
    def __str__(self):
        values = [str(x) for x in self.linkedList]
        return ' '.join(values)
    
    def enqueue(self, value):
        '''This function adds an item to the rear end of the queue '''
        newNode = Node(value)
        if self.linkedList.head == None:
            self.linkedList.head = newNode
            self.linkedList.tail = newNode
        else:
            self.linkedList.tail.next = newNode
            self.linkedList.tail = newNode
    
    def isEmpty(self):
        ''' Return True if the queue is empty, False otherwise '''
        if self.linkedList.head == None:
            return True
        else:
            return False
    
    def dequeue(self):
        ''' This function removes an item from the front end of the queue '''
        if self.isEmpty():
            return "There is not any node in the Queue"
        else:
            tempNode = self.linkedList.head
            if self.linkedList.head == self.linkedList.tail:
                self.linkedList.head = None
                self.linkedList.tail = None
            else:
                self.linkedList.head = self.linkedList.head.next
            return tempNode
    
    def peek(self):
        ''' This function helps to see the first element at the fron end of the queue '''
        if self.isEmpty():
            return "There is not any node in the Queue"
        else:
            return self.linkedList.head
    
    def delete(self):
        ''' This function deletes an item from the front end of the queue'''
        self.linkedList.head = None
        self.linkedList.tail = None

class TreeNode:
    ''' Defining Tree Node with left child and right child'''
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None

def search_bt(rootNode, nodeValue):
    ''' This function search for node value at given(root) node '''
    if not rootNode:
        return "The BT does not exist"
    else:
        customQueue = Queue()
        customQueue.enqueue(rootNode)
        while not(customQueue.isEmpty()):
            root = customQueue.dequeue()
            if root.value.data == nodeValue:
                return "Success"
            if (root.value.leftChild is not None):
                customQueue.enqueue(root.value.leftChild)
            
            if (root.value.rightChild is not None):
                customQueue.enqueue(root.value.rightChild)
        return "Not found"

def insert_node_bt(rootNode, newNode):
     ''' This function insert new node at given(root) node '''
    if not rootNode:
        rootNode = newNode
    else:
        customQueue = Queue()
        customQueue.enqueue(rootNode)
        while not(customQueue.isEmpty()):
            root = customQueue.dequeue()
            if root.value.leftChild is not None:
                customQueue.enqueue(root.value.leftChild)
            else:
                root.value.leftChild = newNode
                return "Successfully Inserted"
            if root.value.rightChild is not None:
                customQueue.enqueue(root.value.rightChild)
            else:
                root.value.rightChild = newNode
                return "Successfully Inserted"

def delete_bt(rootNode):
    ''' This function deletes given node '''
    rootNode.data = None
    rootNode.leftChild = None
    rootNode.rightChild = None
    return "The BT has been successfully deleted" 

class dsad_assignment1:
    ''' Defining DSAD Assignment object'''
    def __init__(self,input_path,output_path,first_node_name='ce',num_oper = 10):
        self.input_path = input_path
        self.output_path = output_path
        self.first_node = TreeNode(first_node_name)
        self.tree_obj = {}
        self.tree_obj.update({first_node_name:self.first_node})
        with open(self.input_path,mode='r',encoding="utf-8") as f:
            self.contents = f.readlines()
        self.outputfile = open(output_path, 'w+')
        self.company_track = defaultdict(list)
        self.num_oper = num_oper
        self.all_oper = 0

    def detail(self,company_name):
        ''' printing the details of the company from the linked list '''
        self.print(f"DETAIL:{company_name}")
        childlist = self.company_track[company_name]
        self.print(f"Acquired companies: {','.join(childlist) if len(childlist) > 0 else 'none'}")
        self.print(f"No of companies acquired: {len(childlist) }")
        self.all_oper = self.all_oper + 1

    def acquire(self,parent_company,acquired_company):
        ''' Acquiring/Updating the details of the company in the linked list '''
        if search_bt(self.first_node, acquired_company) != 'Success':
            acqobj = TreeNode(acquired_company)
            self.tree_obj.update({acquired_company:acqobj})
            insert_node_bt(self.tree_obj[parent_company], acqobj)
            self.print(f"ACQUIRED SUCCESS:{parent_company} Successfully acquired {acquired_company}")
            self.company_track[parent_company].append(acquired_company)
        else:
            self.print(f"ACQUIRED FAILED: {acquired_company} BY:{parent_company}")
        self.all_oper = self.all_oper + 1
    
    def release(self,released_company):
        ''' Releasing / Removing the details of the company in the linked list '''
        if search_bt(self.first_node, released_company) == 'Success':
            delete_bt(self.tree_obj[released_company])
            self.print(f"RELEASED SUCCESS: released {released_company} successfully.")
            for key,items in dict(self.company_track).items():
                if released_company in items:
                    self.company_track[key].remove(released_company)
        else:
            self.print(f"RELEASED FAILED: released {released_company} failed.")
        self.all_oper = self.all_oper + 1

    def print(self,verbose):
        ''' printing output file '''
        print(verbose)
        self.outputfile.write(f"{verbose}\n")

    def run_job(self):
        ''' Run job : this function helps to read the file and preform given action
        like acquire, details and release operation and writes output to output file 
        '''
        for content in self.contents:
            action = content.split(':')[0].split(' ')[0].strip()
            if self.all_oper < self.num_oper:
                if action == 'DETAIL':
                    company_name = content.split(':')[0].split(' ')[1].strip()
                    self.detail(company_name)
                elif action == 'ACQUIRED':
                    acquired_company = content.split(':')[1].split(' ')[0].strip()
                    parent_company = content.split(':')[-1].strip()
                    self.acquire(parent_company,acquired_company)
                elif  action == 'RELEASE':
                    released_company = content.split(':')[0].split(' ')[1].strip()
                    self.release(released_company)
                elif action == 'Company':
                    first_node_name = content.split(':')[1].strip()
                    self.first_node = TreeNode(first_node_name)
                elif action == 'No':
                    num_oper = content.split(':')[1].strip()
                    self.num_oper = int(num_oper)
                else:
                    print(f"Unknown action : {action} mentioned in the file")
        self.outputfile.close()

if __name__ == "__main__":
#     Defining input , output file path and initializing dsad_assignment1 object and calling run_job function to preform given action
    input_path = r".\\inputPS5.txt"
    output_path = r".\\outputPS5.txt"
    da = dsad_assignment1(input_path,output_path,first_node_name='ce',num_oper=100)
    da.run_job()
