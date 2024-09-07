// Author : Tin Le

#pragma GCC target("popcnt")
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ub upper_bound
#define lb lower_bound
#define db double
#define ll unsigned long long
#define int long long
#define vi vector<int>
#define pii pair<int, int>
#define vpii vector<pair<int, int>>
#define vd vector<db>
#define ar(x) array<int, x>

#define pb push_back
#define f first
#define s second
#define rsz resize
#define sum(x) accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define rsrt(x) sort(allr(x))
#define sortErase(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define reverse(x) reverse(all(x))

const static int INF = 1LL << 61;
const static int MX = 2e5 + 5;
const static int MOD = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
constexpr int pct(int x) { return __builtin_popcount(x); }

const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

constexpr int modExpo(int base, int exp, int mod)
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
const int MAX = 1 << 21;
int dp1[MAX], dp2[MAX];
void solve()
{
    int n; cin >> n;
    vi arr(n);
    for(auto& it : arr)
    {
        cin >> it;
        dp1[it]++, dp2[it]++;
    }

    for(int k = 0; k < 21; k++)
    {
        for(int i = 0; i < MAX; i++)
        {
            if((i >> k) & 1) dp1[i] += dp1[i ^ (1 << k)];
        }
    }
    for(int k = 0; k < 21; k++)
    {
        for(int i = MAX - 1; i >= 0; i--)
        {
            if((i >> k) & 1) dp2[i ^ (1 << k)] += dp2[i];
        }
    }
    for(auto& it : arr)
    {
        int ans1 = dp1[it];
        int ans2 = dp2[it];
        int ans3 = n - dp1[it ^ (MAX - 1)];
        cout << ans1 << " " << ans2 << " " << ans3 << endl;
    }

}

signed main()
{
    ios::sync_with_stdio(false); cin.tie(nullptr); 
    solve();
    return 0;
}

