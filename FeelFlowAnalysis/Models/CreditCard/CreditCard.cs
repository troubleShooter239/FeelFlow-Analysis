namespace FeelFlowAnalysis.Models;

public struct CreditCard : ICreditCard
{
    public string CardOwner { get; set; }
    public int CardNumber { get; set; }
    public int CVV { get; set; }
    public int ExpMonth { get; set; }
    public int ExpYear { get; set; }
}
