import torch


class Solution:
    def shortestPalindrome(self, s: str) -> str:
        N = len(s)
        if N < 2 : return s
        res = 0
        for i in range(1,N):
            m = (i+1)//2
            if s[:m] == s[m if (i+1)%2==0 else m+1:i+1][::-1] : res = i
        return s[res+1:][::-1]+s


if __name__ == "__main__":
    a = Solution()
    s = "helloword"
    b = a.shortestPalindrome(s)
    print(b)
    d = [1, 2, 3, 4]
    e = torch.tensor(d)
    print(d)

