#include <bits/stdc++.h>
using namespace std;
 
// Test-case generator for the UNION/GET sets problem.
// This version does not read any input: it randomly chooses n and q.
 
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
 
    // Use a random number generator seeded with the current time.
    mt19937 rng((unsigned) chrono::steady_clock::now().time_since_epoch().count());
    
    // Choose n and q randomly within desired ranges.
    // For example, n between 1 and 50, q between 1 and 100.
    int n = rng() % 5 + 1;      // initial number of sets
    int q = rng() % 5 + 1;     // total number of queries to generate
 
    // We will have at most n+q sets because each UNION produces a new set.
    vector<vector<int>> sets(n + q + 1);
    vector<int> sz(n + q + 1, 0); // size of each set
    vector<int> active;           // active set ids (those that haven't been merged away)
    
    // Initialize the n sets; each set i contains just the element i.
    for (int i = 1; i <= n; i++){
        sets[i].push_back(i);
        sz[i] = 1;
        active.push_back(i);
    }
    
    // curID tracks the highest set id used so far.
    int curID = n;
    
    // We'll store the generated queries in a vector of strings.
    vector<string> queries;
    queries.reserve(q);
 
    // Generate q queries.
    for (int i = 0; i < q; i++){
        bool doUnion = false;
        // We can perform a UNION only if there are at least two active sets.
        if (active.size() >= 2)
            doUnion = (rng() % 2 == 0); // roughly 50% chance
        // Otherwise, we must do a GET.
 
        if (doUnion) {
            // Pick two distinct active sets at random.
            int idx1 = rng() % active.size();
            int idx2 = rng() % active.size();
            while (idx2 == idx1)
                idx2 = rng() % active.size();
 
            int a = active[idx1];
            int b = active[idx2];
 
            // Output a UNION query.
            queries.push_back("UNION " + to_string(a) + " " + to_string(b));
 
            // Simulate the union: merge the two sorted sets.
            vector<int> merged;
            int i1 = 0, i2 = 0;
            while (i1 < sets[a].size() && i2 < sets[b].size()){
                if (sets[a][i1] < sets[b][i2]) {
                    merged.push_back(sets[a][i1]);
                    i1++;
                } else {
                    merged.push_back(sets[b][i2]);
                    i2++;
                }
            }
            while (i1 < sets[a].size()){
                merged.push_back(sets[a][i1]);
                i1++;
            }
            while (i2 < sets[b].size()){
                merged.push_back(sets[b][i2]);
                i2++;
            }
 
            // Create a new set with a new id.
            curID++;
            sets[curID] = merged;
            sz[curID] = sz[a] + sz[b];
 
            // Remove the merged sets from the active list.
            // Erase the larger index first to avoid shifting issues.
            if (idx1 > idx2) swap(idx1, idx2);
            active.erase(active.begin() + idx2);
            active.erase(active.begin() + idx1);
 
            // Add the new set id.
            active.push_back(curID);
        } else {
            // GET query: choose one active set at random.
            int idx = rng() % active.size();
            int a = active[idx];
            int setSize = sz[a];
            // Choose k uniformly in [1, setSize].
            int k_val = (rng() % setSize) + 1;
            queries.push_back("GET " + to_string(a) + " " + to_string(k_val));
        }
    }
 
    // Output the test case.
    // First line: initial n and q.
    cout << n << " " << q << "\n";
    // Then each query on its own line.
    for (auto &s : queries) {
        cout << s << "\n";
    }
 
    return 0;
}

