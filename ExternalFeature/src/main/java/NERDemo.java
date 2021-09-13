import java.util.List;

public class NERDemo {

    public static void main(String[] args) {

        String[] example = {"Good afternoon Rajat Raina, how are you today?",
                "I go to school at Stanford University, which is located in California.",
                "Laboratoire de Physique Subatomique et de Cosmologie, Université Joseph Fourier and CNRS/IN2P3 and Institut National Polytechnique de Grenoble, Grenoble, France.",
                "DSM/IRFU (Institut de Recherches sur les Lois Fondamentales de l'Univers), CEA Saclay (Commissariat à l'Energie Atomique et aux Energies Alternatives), Gif-sur-Yvette, France.",
                "Institute of Nuclear and Particle Physics (INPP), NCSR Demokritos, Aghia Paraskevi, Greece.",
                "Physics Department, Brookhaven National Laboratory, Upton, NY United States of America."
        };

        Object[] objects = NerUtil.parseOrganizationLocation(example[3]);
        List<String> locList = (List<String>) objects[0];
        List<String> orgList = (List<String>) objects[1];
        for (String s : orgList) {
            System.out.println(s);
        }

        System.out.println();
        for (String s : locList) {
            System.out.println(s);
        }
    }

}