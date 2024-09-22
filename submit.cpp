#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

// Function to apply the operation on a binary string
string apply_operation(const string &C, int M) {
    string new_C(M, '0');
    int sum = 0;
    for(int k=0; k<M; ++k){
        sum = (sum + (C[k]-'0')) % 2;
        new_C[k] = '0' + sum;
    }
    return new_C;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    int N, M;
    cin >> N >> M;
    
    vector<string> sequences(N, string(M, '0'));
    for(int i=0; i<N; ++i){
        for(int k=0; k<M; ++k){
            char c;
            cin >> c;
            sequences[i][k] = c;
        }
    }
    
    // Initialize f as a 1D array, size N*N
    // f[i*N +j] stores f(i,j)
    // Initialize to -1
    // But since N*N can be up to 1e6, use a vector of int
    vector<int> f(N*N, -1);
    // f(i,j) is stored at f[i*N +j]
    
    // Initialize: for all i, f[i][i}=0
    for(int i=0; i<N; ++i){
        f[i*N +i] = 0;
    }
    
    // Initialize current states
    vector<string> current_states = sequences;
    
    // Maximum steps to prevent infinite loops, set to M+1
    // But can be up to log2(M)
    int max_steps = M + 1;
    
    for(int x=0; x<=max_steps; ++x){
        // Group sequences by their current state
        // Use a map from string to vector of indices
        // To speed up, use a hash map with string keys
        // But with M up to 1000 and N up to 1000, manageable
        unordered_map<string, vector<int>> groups;
        for(int i=0; i<N; ++i){
            groups[current_states[i]].push_back(i);
        }
        
        // For each group, assign f[i][j}=x for all pairs if not set
        for(auto &[key, vec]: groups){
            if(vec.size() >=2){
                // Assign f[i][j} =x for all pairs in vec
                for(int i=0; i<vec.size(); ++i){
                    for(int j=i+1; j<vec.size(); ++j){
                        int a = vec[i];
                        int b = vec[j];
                        if(f[a*N +b] == -1){
                            f[a*N +b] = x;
                            f[b*N +a] = x;
                        }
                    }
                }
            }
        }
        
        // If x == max_steps, break
        if(x == max_steps){
            break;
        }
        
        // Compute next state
        vector<string> next_states(N, string(M, '0'));
        for(int i=0; i<N; ++i){
            next_states[i] = apply_operation(current_states[i], M);
        }
        current_states = next_states;
    }
    
    // Now, compute the sum
    ll MOD = 998244353;
    ll total_sum =0;
    for(int i=0; i<N; ++i){
        for(int j=i; j<N; ++j){
            if(i ==j){
                total_sum +=0;
            }
            else{
                if(f[i*N +j] != -1){
                    total_sum +=f[i*N +j];
                }
                else{
                    // f(i,j}=0
                    // According to the problem statement, add 0
                }
            }
            if(total_sum >= MOD){
                total_sum -= MOD;
            }
        }
    }
    
    cout << total_sum % MOD;
}

