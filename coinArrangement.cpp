// Author : Tin Le
//   __________________
//  | ________________ |
//  ||          ____  ||
//  ||   /\    |      ||
//  ||  /__\   |      ||
//  || /    \  |____  ||
//  ||________________||
//  |__________________|
//  \###################\
//   \###################\
//    \        ____       \
//     \_______\___\_______\
// An AC a day keeps the doctor away.

#pragma GCC optimize("Ofast")
#pragma GCC optimize ("unroll-loops")
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
#define vvi vector<vi>
#define vd vector<db>
#define ar(x) array<int, x>
#define mset(m, v) memset(m, v, sizeof(m))
#define pb push_back
#define f first
#define s second
#define rsz resize
#define sum(x) accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define rev(x) reverse(all(x))
#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

const static int INF = 1LL << 61;
const static int MX = 2e5 + 5;
const static int MOD = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
constexpr int pct(int x) { return __builtin_popcount(x); }
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
constexpr int modExpo(int base, int exp, int mod) { int res = 1; while(exp) {
    if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>=
        1; } return res; }
int arr[2][MX];
void solve()
{
    int n; cin >> n;
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < n; j++) cin >> arr[i][j], arr[i][j]--;
    }

    int res = 0, bot = 0, top = 0;
    for(int i = 0; i < n; i++)
    {
        top += arr[0][i], bot += arr[1][i];
        if((top > 0 && bot < 0) || (top < 0 && bot > 0))
        {
            if(abs(top) < abs(bot))
            {
                res += abs(top);
                bot += top;
                top = 0;
            }
            else
            {
                res += abs(bot);
                top += bot;
                bot = 0;
            }
        }
        res += abs(top) + abs(bot);
    }
    cout << res << endl;
}

signed main()
{
    IOS;
    int t = 1;
    // cin >> t;
    while(t--) solve();
    #ifdef LOCAL
    clock_t tStart = clock();
    cout<<fixed<<setprecision(10)<<"\nTime Taken: "<<(double)(clock()- tStart)/CLOCKS_PER_SEC<<endl;
    #endif
    return 0;
}

