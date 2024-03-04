namespace FeelFlowAnalysis.Models;

public interface IUser
{
    string Id { get; set; }
    string Email { get; set; }
    string PasswordHash { get; set; }
    CreditCard CardDetails { get; set; }
}
