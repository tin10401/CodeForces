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
#define sort(x) sort(all(x))
#define sortr(x) sort(allr(x))
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

void solve()
{
    int n, k; cin >> n >> k;
    deque<int> q;
    for(int i = 1; i <= n; i++) q.pb(i);
    vi res(n + 1);
    int index = 1;
    for(int i = n - 1; i >= 0; i--)
    {
        if(k >= i)
        {
            res[index++] = q.back(); q.pop_back();
            k -= i;
        }
        else 
        {
            res[index++] = q.front(); q.pop_front();
        }
    }
    for(int i = 1; i <= n; i++) cout << res[i] << " ";
    cout << endl;
}

signed main()
{
    ios::sync_with_stdio(false); cin.tie(nullptr); 
    solve();
    return 0;
}

