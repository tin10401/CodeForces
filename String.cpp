class Binary_Trie { 
    public:
    int T[MX][2];   
    int ptr;    
    Binary_Trie() {    
        ptr = 0;    
        mset(T, 0);
    }
    
    void insert(int num) {  
        int curr = 0;   
        for(int i = 31; i >= 0; i--) {  
            int bits = (num >> i) & 1;  
            if(!T[curr][bits]) T[curr][bits] = ++ptr;   
            curr = T[curr][bits];
        }
    }
        
    int max_xor(int num) {  
        int res = 0, curr = 0;
        for(int i = 31; i >= 0; i--) {  
            int bits = (num >> i) & 1;  
            if(T[curr][!bits]) {    
                curr = T[curr][!bits];
                res |= (1LL << i);
            }
            else {  
                curr = T[curr][bits];
            }
        }
        return res;
    }
};

struct TrieNode
{
    TrieNode* sfx, *dict, *children[26];
    int id = -1;
};

static TrieNode nodes[(int)5e4 + 1];

class Trie
{
    public:
    TrieNode* root;
    int count = 0;
    TrieNode* newTrieNode() {
        nodes[count] = TrieNode();
        return &nodes[count++];
    }

    Trie(vector<string>& words, vector<int>& costs, string target) {
        root = newTrieNode();
        root->sfx = root->dict = root;

    }
    
    void insert(const string& s) {  
        TrieNode* curr = root;
        for(auto& ch : s) {
            if(!curr->children[ch - 'a']) curr->children[ch - 'a'] = newTrieNode();
            curr = curr->children[ch - 'a'];
        }

    }
    
    void aho_corasick() {   
        queue<TrieNode*> q;
        q.push(root);
        while(!q.empty()) {
            TrieNode* par = q.front();
            q.pop();
            for(int i = 0; i < 26; i++) {
                TrieNode* child = par->children[i];
                if(!child) continue;
                TrieNode* suff = par->sfx;
                while(suff != root && !suff->children[i]) suff = suff->sfx;
                if(par != root && suff->children[i]) child->sfx = suff->children[i];
                else child->sfx = root;

                child->dict = child->sfx->id == -1 ? child->sfx->dict : child->sfx;
                q.push(child);
            }
        }
    }

    void queries(TrieNode*& prev, int i, char ch)
    {
        while(prev != root && !prev->children[ch - 'a']) prev = prev->sfx;
        if(prev->children[ch - 'a']) {
            prev = prev->children[ch - 'a'];
            TrieNode* curr = prev->id == -1 ? prev->dict : prev;
            while(curr->id != -1){
                int j = curr->id;
                curr = curr->dict;
            }
        }
    }
};

vi KMP(const string& s) {   
    int n = s.size();
    vi prefix(n);
    for(int i = 1, j = 0; i < n; i++) { 
        while(j && s[i] != s[j]) j = prefix[j - 1]; 
        if(s[i] == s[j]) prefix[i] = ++j;
    }
    return prefix;
}

vi Z_Function(const string& s) {    
    int n = s.size();   
    vi prefix(n);   
    for(int i = 1, left = 0, right = 0; i < n; i++) {   
        if(i > right) { 
            left = right = i;   
            while(right < n && s[right] == s[right - left]) right++;    
            prefix[i] = right-- - left;
        }
        else {  
            if(prefix[i - left] + i < right + 1) {  
                prefix[i] = prefix[i - left];
            }
            else {  
                left = i;   
                while(right < n && s[right] == s[right - left]) right++;    
                prefix[i] = right-- - left;
            }
        }
    }
    return prefix;
}
    
vi manacher(string s, int start) {
    string tmp;
    for (auto& it : s) {
        tmp += "#";
        tmp += it;
    }
    tmp += "#";  
    swap(s, tmp);
    int n = s.size();
    vector<int> p(n); 
    int l = 0, r = 0;  
    for (int i = 0; i < n; i++) {
        if (i < r) {
            p[i] = min(r - i, p[l + r - i]);
        } else {
            p[i] = 0;
        }
        while (i - p[i] >= 0 && i + p[i] < n && s[i - p[i]] == s[i + p[i]]) {
            p[i]++;
        }
        if (i + p[i] > r) {
            l = i - p[i] + 1;
            r = i + p[i] - 1;
        }
    }
    vi result;
    for (int i = start; i < n; i += 2) {
        result.push_back(p[i] / 2);
    }
    return result;
}

class RabinKarp {   
    public: 
    vvll prefix, pow;
    vll base, mod;
    int n, m;
    RabinKarp(const string& s) {  
        m = 3;
        base = {26, 28, 30};    
        mod = {(int)1e9 + 7, (int)1e9 + 33, (int)1e9 + 73};
        n = s.size(); 
        pow.rsz(m), prefix.rsz(m); 
        for(int i = 0; i < m; i++) {    
            pow[i].rsz(n + 1, 1);  
            prefix[i].rsz(n + 1);
        }
        buildHash(s);
    }
    
    void buildHash(const string& s) {   
        for(int j = 1; j <= n; j++) {   
            int x = s[j - 1] - 'a' + 1;
            for(int i = 0; i < m; i++) {    
                prefix[i][j] = (prefix[i][j - 1] * base[i] + x) % mod[i];   
                pow[i][j] = (pow[i][j - 1] * base[i]) % mod[i];
            }
        }
    }
    
    int getHash(int l, int r) { 
		if(l < 0 || r > n || l > r) return -1;
        int hash = prefix[0][r] - (prefix[0][l] * pow[0][r - l] % mod[0]) % mod[0];
        hash = (hash + mod[0]) % mod[0];
        return hash;
		
//        if(l < 0 || r > n || l > r) return {-1, -1};
//        vll ans;    
//        for(int i = 0; i < m; i++) {    
//            ll hash = prefix[i][r] - (prefix[i][l] * pow[i][r - l] % mod[i]) % mod[i]; 
//            hash = (hash + mod[i]) % mod[i];
//            ans.pb(hash);
//        }
//        return MP(ans[0], ans[1]);
    };

    
	bool diff_by_one_char(RabinKarp& a, int offSet = 0) { // a.size() > n
        int left = 0, right = n, rightMost = -1;    
        while(left <= right) {  
            int middle = midPoint;  
            if(a.getHash(offSet, middle + offSet) == getHash(0, middle)) rightMost = middle, left = middle + 1; 
            else right = middle - 1;
        }
        return a.getHash(rightMost + 1 + offSet, offSet + n) == getHash(rightMost + 1, n);
    }

};

