#include <bits.stdc++.h>
using namespace std;

struct Node {
    Node* next;
    int val;
    Node(int val = 0) : val(val), next(nullptr) {}
};
class Linked_list {
    public:
    Node* head;
    Linked_list() {}

    ~Linked_list() {
        while(head) {
            Node* curr = head;
            head = head->next;
            delete curr;
        }
    }
    void insert(int val) {
        Node* curr = head;
        while(curr) {
            curr = curr->next;
        }
        curr = new Node(val);
    }

    void delete_front() {
        Node* curr = head;
        head = head->next;
        delete curr;
    }
};
int main() {

}
