#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    // Set seed for random generator
    srand((unsigned) time(0));
    
    // You can modify these values or take them as input
    int n = 10; // number of DSU elements
    int q = 30; // number of queries

    cout << n << " " << q << "\n";
    
    for (int i = 1; i <= q; i++) {
        int op = rand() % 2; // randomly choose op: 0 for merge, 1 for query
        
        // For valid persistent DSU operations, we want k (version parameter)
        // to be in the range [0, i-1]. For the first query, k will be 0.
        int k = (i == 1 ? 0 : rand() % i);
        
        // Choose random nodes u and v from 0 to n-1.
        int u = rand() % n;
        int v = rand() % n;
        
        // For merge queries ensure that u and v are distinct.
        if (op == 0 && u == v) {
            v = (v + 1) % n;
        }
        
        cout << op << " " << k << " " << u << " " << v << "\n";
    }
    
    return 0;
}

