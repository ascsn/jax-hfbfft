"""
Tests for the Nucleus class.
"""

import pytest
from jax_hfbfft.core.nucleus import Nucleus


class TestNucleus:
    """Tests for Nucleus class."""
    
    def test_basic_creation(self):
        """Test basic nucleus creation."""
        nucleus = Nucleus(protons=50, neutrons=82)
        
        assert nucleus.protons == 50
        assert nucleus.neutrons == 82
        assert nucleus.Z == 50
        assert nucleus.N == 82
        assert nucleus.mass_number == 132
        assert nucleus.A == 132
    
    def test_creation_with_name(self):
        """Test nucleus creation with a name."""
        nucleus = Nucleus(protons=50, neutrons=82, name="Sn-132")
        
        assert nucleus.name == "Sn-132"
        assert str(nucleus) == "Sn-132"
    
    def test_from_symbol(self):
        """Test creation from element symbol."""
        ca40 = Nucleus.from_symbol("Ca", 40)
        
        assert ca40.protons == 20
        assert ca40.neutrons == 20
        assert ca40.mass_number == 40
        assert ca40.name == "Ca-40"
    
    def test_from_symbol_sn132(self):
        """Test creation of Sn-132 from symbol."""
        sn132 = Nucleus.from_symbol("Sn", 132)
        
        assert sn132.protons == 50
        assert sn132.neutrons == 82
    
    def test_from_symbol_pb208(self):
        """Test creation of Pb-208 from symbol."""
        pb208 = Nucleus.from_symbol("Pb", 208)
        
        assert pb208.protons == 82
        assert pb208.neutrons == 126
    
    def test_invalid_protons(self):
        """Test that negative proton number raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            Nucleus(protons=-1, neutrons=10)
    
    def test_invalid_neutrons(self):
        """Test that negative neutron number raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            Nucleus(protons=10, neutrons=-1)
    
    def test_unknown_element(self):
        """Test that unknown element symbol raises error."""
        with pytest.raises(ValueError, match="Unknown element"):
            Nucleus.from_symbol("Xx", 100)
    
    def test_invalid_mass_number(self):
        """Test that invalid mass number raises error."""
        with pytest.raises(ValueError, match="negative neutron"):
            Nucleus.from_symbol("Ca", 10)  # Would give N = -10
    
    def test_immutability(self):
        """Test that Nucleus is immutable (frozen dataclass)."""
        nucleus = Nucleus(protons=50, neutrons=82)
        
        with pytest.raises(Exception):  # FrozenInstanceError
            nucleus.protons = 51
    
    def test_repr(self):
        """Test string representation."""
        nucleus = Nucleus(protons=50, neutrons=82, name="Sn-132")
        
        repr_str = repr(nucleus)
        assert "Nucleus" in repr_str
        assert "protons=50" in repr_str
        assert "neutrons=82" in repr_str


class TestNucleusEquality:
    """Tests for Nucleus equality comparisons."""
    
    def test_equal_nuclei(self):
        """Test that identical nuclei are equal."""
        n1 = Nucleus(protons=50, neutrons=82)
        n2 = Nucleus(protons=50, neutrons=82)
        
        assert n1 == n2
    
    def test_different_nuclei(self):
        """Test that different nuclei are not equal."""
        n1 = Nucleus(protons=50, neutrons=82)
        n2 = Nucleus(protons=50, neutrons=80)
        
        assert n1 != n2
    
    def test_same_with_different_names(self):
        """Test that same Z,N with different names are still equal."""
        n1 = Nucleus(protons=50, neutrons=82, name="Sn-132")
        n2 = Nucleus(protons=50, neutrons=82, name="Tin-132")
        
        # Names differ, so they should not be equal with frozen dataclass default
        assert n1 != n2
