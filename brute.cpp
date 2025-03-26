#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
using namespace std;
 
typedef long long ll;
 
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int T;
    cin >> T;
    while(T--){
        int n;
        cin >> n;
        vector<ll> capacities(n);
        ll totalSum = 0;
        for (int i = 0; i < n; i++){
            cin >> capacities[i];
            totalSum += capacities[i];
        }
        int q; cin >> q;
        
        // Process each mission
        for (int j = 0; j < q; j++){
            ll X, Y;
            cin >> X >> Y;
            ll ans = LLONG_MAX;
            
            // Try each cabin as the crew cabin.
            for (int i = 0; i < n; i++){
                // Tokens needed to satisfy crew requirement for cabin i:
                ll crewTokens = max(0LL, X - capacities[i]);
                // Tokens needed to satisfy storage requirement for other cabins:
                ll storageTokens = max(0LL, Y - (totalSum - capacities[i]));
                ans = min(ans, crewTokens + storageTokens);
            }
            
            cout << ans << "\n";
        }
    }
    
    return 0;
}

