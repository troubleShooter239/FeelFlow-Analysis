namespace FeelFlowAnalysis.Models.Entities;

// Summary:
//     Represents a credit card entity.
public struct CreditCard
{
    // Summary:
    //     Gets or sets the name of the card owner.
    public string CardOwner { get; set; }

    // Summary:
    //     Gets or sets the card number.
    public int CardNumber { get; set; }

    // Summary:
    //     Gets or sets the card's CVV.
    public int CVV { get; set; }

    // Summary:
    //     Gets or sets the expiration month of the card.
    public int ExpMonth { get; set; }

    // Summary:
    //     Gets or sets the expiration year of the card.
    public int ExpYear { get; set; }
}
