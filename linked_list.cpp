#include <bits.stdc++.h>
#include <vector>
#include <algorithm>
using namespace std;

const int MX = 1e5;
int storage[MX], ptr, next[MX], val[MX];


class Linked_list {
    public:

    int head = 0;
    Linked_list() {}

    ~Linked_list() {
        for(int i = 0; i <= ptr; i++) {
            storage[i] = 0;
            next[i] = 0;
        }
        ptr = 0;
    }

    void insert(int x) {
        int curr = head;
        while(curr) {
            curr = next[curr];
        }
        storage[curr] = ++ptr;
        val[curr] = x;
    }

    void delete_front() {
        int curr = head;
        head = next[head];
        storage[curr] = 0;
    }
    
    void reverse() {
        if(!head) return;
        int curr = head;
        vector<int> res;
        while(head) {
            res.pb(head);
            head = next[head];
        }
        reverse(begin(res), end(res));
        head = res.begin();
        int n = res.size();
        for(int i = 1; i < n; i++) {
            next[res[i - 1]] = res[i];
        }
    }
};

int main() {

}
