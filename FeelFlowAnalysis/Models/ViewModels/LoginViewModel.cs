using System.ComponentModel.DataAnnotations;

namespace FeelFlowAnalysis.Models.ViewModels;

public class LoginViewModel
{
    [Required(AllowEmptyStrings = false, ErrorMessage = "Please enter email")]
    public string? Email { get; set; }

    [Required(AllowEmptyStrings = false, ErrorMessage = "Please enter password")]
    public string? Password { get; set; }
}
