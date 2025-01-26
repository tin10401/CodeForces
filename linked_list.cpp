#include <bits.stdc++.h>
using namespace std;

struct queue {
    deque<int> s;
    void insert(int x) {
        s.push_back(x);
    }

    bool pop() {
        if(s.empty()) return false;
        s.pop_front();
        return true;
    }

    int front() {
        return s.empty() ? -1 : s.front();
    }

    void push(int x) {
        s.push_back(x);
    }

    int get_size() {
        return s.size();
    }
};

int main() {
    queue s;
    int x; cin >> x;
    s.push(x);
}

