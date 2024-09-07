// https://cses.fi/problemset/task/2417/
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
const static int INF = 1LL << 61;
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

int modExpo(int base, int exp)
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
vi primes;
bitset<1000001> bt;
void generate()
{
    int m = 1e6 + 1;
    bt.set(2);
    for(int i = 3; i < m; i += 2) bt.set(i);
    for(int i = 3; i * i < m; i += 2)
    {
        if(bt[i])
        {
            for(int j = i; j * i < m; j += 2)
            {
                bt.reset(j * i);
            }
        }
    }
    for(int i = 2; i < m; i++) if(bt[i]) primes.pb(i);
}

void solve()
{
    int n;
    cin >> n;
    int res = n * (n - 1) / 2;
    int cnt[1000001] = {};
    generate();
    for(int i = 0; i < n; i++)
    {
        int num;
        cin >> num;
       int x = num;
       vi list;
       for(auto& p : primes)
       {
           if(num == 1) break;
           else if(bt[num]) {list.pb(num); break; }
           if(num % p == 0)
           {
               list.pb(p);
               while(num % p == 0) num /= p;
           }
       }
       int k = list.size();
       for(int mask = 1; mask < 1 << k; mask++)
        {
            int p = 1;
            for(int j = 0; j < k; j++)
            {
                if((mask >> j) & 1)
                {
                    p *= list[j];
                }
            }
            int x = __builtin_popcount(mask) & 1 ? 1 : -1;
            res -= cnt[p]++ * x;
        }
    }
    cout << res << endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

