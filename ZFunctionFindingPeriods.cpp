#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ll unsigned long long
#define int long long
#define pb push_back
#define vi vector<int>
#define pii pair<int, int>
#define vpii vector<pair<int, int>>
#define f first
#define s second
#define ar(x) array<int, x>
const static int INF = 1e18;
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

int modExpo(ll base, ll exp)
{
    int res = 1;
    while(exp)
    {
        if(exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return res;
}

void solve()
{
    string s;
    cin >> s;
    int n = s.size();
    vi prefix(n);
    for(int i = 1, left = 0, right = 0; i < n; i++)
    {
        if(i > right)
        {
            left = right = i;
            while(right < n && s[right] == s[right - left]) right++;
            prefix[i] = right-- - left;
        }
        else
        {
            if(prefix[i - left] + i < right + 1)
            {
                prefix[i] = prefix[i - left];
            }
            else
            {
                left = i;
                while(right < n && s[right] == s[right - left]) right++;
                prefix[i] = right-- - left;
            }
        }
    }

    for(int i = 0; i < n; i++)
    {
        if(i + prefix[i] == n) cout << i << " ";
    }
    cout << n << endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}
