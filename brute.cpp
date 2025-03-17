#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
 
// Check if a vector is alternating (no two consecutive elements are equal)
bool isAlternating(const vector<int>& arr) {
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] == arr[i-1])
            return false;
    }
    return true;
}
 
// Check if Q is a subsequence of P.
bool isSubsequence(const vector<int>& P, const vector<int>& Q) {
    int j = 0;
    for (int i = 0; i < P.size() && j < Q.size(); i++) {
        if (P[i] == Q[j])
            j++;
    }
    return (j == Q.size());
}
 
// Brute force count: count subarrays of A that match Q (i.e. B)
int bruteForceCount(const vector<int>& A, const vector<int>& Q) {
    int N = A.size(), M = Q.size();
    int count = 0;
    
    // Determine if Q is alternating.
    bool Q_is_alternating = isAlternating(Q);
    
    // Iterate over all subarrays of A.
    for (int L = 0; L < N; L++) {
        for (int R = L; R < N; R++) {
            vector<int> P(A.begin() + L, A.begin() + R + 1);
            // The subarray must have at least M elements.
            if (P.size() < M)
                continue;
            
            if (!Q_is_alternating) {
                // For non-alternating Q, the only possibility is an exact match.
                if (P.size() != M)
                    continue;
                if (P == Q)
                    count++;
            } else {
                // When Q is alternating:
                // 1. The subarray must start with Q[0] (no insertion before the first element).
                // 2. It must be alternating.
                // 3. Q must be a subsequence of P.
                if (P[0] != Q[0])
                    continue;
                if (!isAlternating(P))
                    continue;
                if (isSubsequence(P, Q))
                    count++;
            }
        }
    }
    return count;
}
 
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t;
    cin >> t;
    while(t--){
        int N, M;
        cin >> N >> M;
        vector<int> A(N), B(M);
        for (int i = 0; i < N; i++){
            cin >> A[i];
        }
        for (int j = 0; j < M; j++){
            cin >> B[j];
        }
        
        int ans = bruteForceCount(A, B);
        cout << ans << "\n";
    }
    return 0;
}

