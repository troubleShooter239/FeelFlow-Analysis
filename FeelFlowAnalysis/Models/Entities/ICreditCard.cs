namespace FeelFlowAnalysis.Models.Entities;

public interface ICreditCard
{
    string CardOwner { get; set; }
    int CardNumber { get; set; }
    int CVV { get; set; }
    int ExpMonth { get; set; }
    int ExpYear { get; set; }
}
